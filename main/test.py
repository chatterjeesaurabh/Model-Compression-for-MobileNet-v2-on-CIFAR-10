# main/test.py
# Evaluate MobileNet-v2 on CIFAR-10 with optional PTQ or QAT quantization.
# Uses custom quantization code (no PyTorch quantization API).
#
# Flow for PTQ/QAT:
#   1. Load fp32 model (baseline or QAT-trained)
#   2. Swap modules, calibrate activations, freeze scale/zp
#   3. Quantize weights to actual int8 and save packed model to disk
#   4. Load quantized model from disk (int weights + act params)
#   5. Evaluate: each layer dequantizes int8 -> fp32 before MAC
#
# Usage:
#   Baseline:     python -m main.test my_exp
#   PTQ:          python -m main.test my_exp --ptq --weight_bits 8 --act_bits 8
#   QAT eval:     python -m main.test my_exp --qat --weight_bits 8 --act_bits 8
#   Skip layers:  python -m main.test my_exp --ptq --weight_bits 4 --act_bits 8 --skip_layers classifier

import argparse
import os

import torch

from src.mobilenetv2 import mobilenet_v2
from src.quantize import (
    calibrate,
    freeze_all_quant,
    load_quantized_model,
    model_size_bytes_fp32,
    print_compression,
    quantize_model_weights,
    save_quantized_model,
    swap_to_quant_modules,
)
from src.utils import (
    configure_cudnn,
    configure_mlflow,
    generate_run_name,
    prepare_calib_dataloader,
    prepare_dataloaders,
    set_seed,
    test,
)


def parse_arg():
    parser = argparse.ArgumentParser(description="Evaluate MobileNet-v2 on CIFAR-10")
    parser.add_argument("--exp_name", type=str, default='', help="Experiment name (model directory and mlflow run name)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_calib_batch", type=int, default=32,
                        help="Number of calibration batches for activation-range calibration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--num_classes", type=int, default=10)

    # Quantization options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ptq", action="store_true", help="Post-training quantization")
    group.add_argument("--qat", action="store_true",
                       help="Evaluate a QAT-trained checkpoint (from main.train --qat)")

    parser.add_argument("--weight_bits", type=int, default=8,
                        help="Bit-width for weight quantization")
    parser.add_argument("--act_bits", type=int, default=8,
                        help="Bit-width for activation quantization")
    parser.add_argument("--skip_layers", nargs="*", default=[],
                        help="Layer name prefixes to skip quantization on")

    parser.add_argument("--no_mlflow", action="store_true", help="Disable mlflow logging")
    return parser.parse_args()


def main():
    args = parse_arg()

    set_seed(args.seed)
    configure_cudnn(deterministic=True, benchmark=False)

    exp_name = args.exp_name
    exp_dir = os.path.join(args.model_dir, exp_name)

    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.ptq:
        mode = "ptq"
    elif args.qat:
        mode = "qat"
    else:
        mode = "baseline"
    print(f"Mode: {mode}")

    # Prepare data
    print("Preparing dataset...")
    _, test_dataloader = prepare_dataloaders(args.batch_size)

    # Load fp32 model
    print("Loading model...")
    model = mobilenet_v2(num_classes=args.num_classes)

    if args.qat:
        model_path = os.path.join(exp_dir, "best_model_qat.pth")
    else:
        model_path = os.path.join(exp_dir, "best_model.pth")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded fp32 model from {model_path}")

    fp32_size = model_size_bytes_fp32(model)
    print(f"FP32 model size: {fp32_size / 1024 / 1024:.2f} MB")

    skip_set = set(args.skip_layers)

    if args.ptq or args.qat:
        # ---- Step 1: Swap to quantized modules ----
        print(f"\nApplying quantization: weight_bits={args.weight_bits}, act_bits={args.act_bits}")
        if skip_set:
            print(f"  Skipping layers: {skip_set}")

        swap_to_quant_modules(
            model, weight_bits=args.weight_bits, act_bits=args.act_bits,
            qat=False, skip_layers=skip_set
        )
        model.to(device)

        # ---- Step 2: Calibrate activation ranges ----
        print("Calibrating activations...")
        calib_dataloader = prepare_calib_dataloader(args.batch_size)
        calibrate(model, calib_dataloader, args.n_calib_batch, device=device)

        # ---- Step 3: Freeze scale/zp, quantize weights to int8 ----
        freeze_all_quant(model)
        quantize_model_weights(model)
        print("Weights quantized to int8.")

        # ---- Step 4: Save packed quantized model to disk ----
        quant_filename = f"quantized_{mode}_w{args.weight_bits}_a{args.act_bits}.pth"
        quant_path = os.path.join(exp_dir, quant_filename)
        save_quantized_model(model, quant_path, args.weight_bits, args.act_bits)

        # ---- Step 5: Load quantized model from disk ----
        # Create fresh model with quant modules, then load packed int weights
        model_eval = mobilenet_v2(num_classes=args.num_classes)
        swap_to_quant_modules(
            model_eval, weight_bits=args.weight_bits, act_bits=args.act_bits,
            qat=False, skip_layers=skip_set
        )
        load_quantized_model(quant_path, model_eval)
        model_eval.to(device)

        # Use the loaded model for evaluation
        model = model_eval

    else:
        # Baseline: just move to device
        model.to(device)

    # Warmup
    print("\nWarming up...")
    t = torch.zeros([1, 3, 32, 32], device=device)
    for _ in range(10):
        model(t)

    # ---- Step 6: Evaluate ----
    print("Running evaluation...")
    accuracy = test(model, device, test_dataloader)
    print(f"Accuracy ({mode}): {accuracy:.4f}")

    # Compression summary
    comp_stats = None
    if args.ptq or args.qat:
        example_input = torch.randn(1, 3, 32, 32, device=device)
        comp_stats = print_compression(
            model, weight_bits=args.weight_bits, act_bits=args.act_bits,
            input_tensor=example_input
        )
        quant_file_size = os.path.getsize(quant_path) / 1024 / 1024
        print(f"Quantized file on disk: {quant_file_size:.4f} MB")

    # MLflow logging
    if not args.no_mlflow:
        import mlflow

        run_name = exp_name or generate_run_name("test", mode, 0, args.batch_size, 0)
        configure_mlflow(
            experiment_name="cs6886_mobilenetv2_quantization",
            run_name=run_name,
            config={
                "exp_name": exp_name,
                "mode": mode,
                "weight_bits": args.weight_bits if (args.ptq or args.qat) else 32,
                "act_bits": args.act_bits if (args.ptq or args.qat) else 32,
                "skip_layers": str(list(skip_set)),
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
        )

        log_metrics = {
            "accuracy": accuracy,
            "fp32_model_size_mb": fp32_size / 1024 / 1024,
        }
        if comp_stats is not None:
            log_metrics.update({
                "weight_bits": args.weight_bits,
                "act_bits": args.act_bits,
                "quant_model_size_mb": comp_stats["quant_model_size_mb"],
                "weight_compression_ratio": comp_stats["weight_compression_ratio"],
                "overhead_kb": comp_stats["overhead_kb"],
                "quant_file_size_mb": quant_file_size,
            })
            if "activation_compression_ratio" in comp_stats:
                log_metrics["activation_compression_ratio"] = comp_stats["activation_compression_ratio"]
        mlflow.log_metrics(log_metrics)
        mlflow.end_run()

    print("Done.")


if __name__ == "__main__":
    main()
