# main/sensitivity.py
# Per-layer sensitivity analysis for MobileNet-v2 quantization.
# Tests the effect of quantizing each layer individually (weights and activations
# separately) at various bit-widths and measures accuracy degradation.
#
# Usage:
#   python -m main.sensitivity baseline_run_1_e200_b128_l01 --bit_widths 2 4 8 --n_calib_batch 32

import argparse
import copy
import os

import torch
import torch.nn as nn

from src.mobilenetv2 import mobilenet_v2
from src.quantize import (
    ActFakeQuant,
    QuantConv2d,
    QuantLinear,
    calibrate,
    freeze_all_quant,
    get_quantizable_layers,
    swap_single_layer_quant,
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
    parser = argparse.ArgumentParser(description="Per-layer sensitivity analysis")
    parser.add_argument("--model_dir", type=str,
                        help="Path to experiment directory containing best_model.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_calib_batch", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--bit_widths", type=int, nargs="+", default=[2, 4, 8],
                        help="Bit-widths to test for each layer")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional experiment name for mlflow run (defaults to model_dir basename)")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable mlflow logging")
    return parser.parse_args()


def evaluate_single_layer_weight_quant(base_model, layer_name, weight_bits,
                                       calib_dataloader, test_dataloader,
                                       n_calib_batch, device):
    """
    Quantize only a single layer's weights and measure accuracy.
    All other layers remain in FP32.
    """
    model = copy.deepcopy(base_model)
    swap_single_layer_quant(model, layer_name, weight_bits=weight_bits)
    model.to(device)

    # Freeze the quantized weight layer
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()

    accuracy = test(model, device, test_dataloader)
    del model
    torch.cuda.empty_cache()
    return accuracy


def evaluate_single_layer_act_quant(base_model, layer_name, act_bits,
                                     calib_dataloader, test_dataloader,
                                     n_calib_batch, device):
    """
    Quantize only a single layer's activation and measure accuracy.
    All other layers remain in FP32.
    """
    model = copy.deepcopy(base_model)
    swap_single_layer_quant(model, layer_name, act_bits=act_bits)
    model.to(device)

    # Calibrate the single activation quantizer
    calibrate(model, calib_dataloader, n_calib_batch, device=device)

    # Freeze the activation quantizer
    for mod in model.modules():
        if isinstance(mod, ActFakeQuant):
            mod.freeze()

    accuracy = test(model, device, test_dataloader)
    del model
    torch.cuda.empty_cache()
    return accuracy


def main():
    args = parse_arg()

    set_seed(args.seed)
    configure_cudnn(deterministic=True, benchmark=False)

    exp_dir = args.model_dir
    exp_name = args.exp_name or os.path.basename(os.path.normpath(exp_dir))

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Prepare data
    print("Preparing dataset...")
    _, test_dataloader = prepare_dataloaders(args.batch_size)
    calib_dataloader = prepare_calib_dataloader(args.batch_size)

    # Load baseline model
    print("Loading baseline model...")
    model = mobilenet_v2(num_classes=args.num_classes)
    model_path = os.path.join(exp_dir, "best_model.pth")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Get baseline accuracy
    model.to(device)
    baseline_acc = test(model, device, test_dataloader)
    model.cpu()
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Get list of quantizable layers
    weight_layers, act_layers = get_quantizable_layers(model)
    print(f"\nFound {len(weight_layers)} weight layers and {len(act_layers)} activation layers")

    # Initialize mlflow
    if not args.no_mlflow:
        import mlflow

        run_name = exp_name or generate_run_name("sensitivity", "analysis", 0, args.batch_size, 0)
        configure_mlflow(
            experiment_name="cs6886_mobilenetv2_quantization",
            run_name=f"{run_name}_sensitivity",
            config={
                "exp_name": exp_name,
                "mode": "sensitivity_analysis",
                "bit_widths": str(args.bit_widths),
                "baseline_accuracy": baseline_acc,
                "seed": args.seed,
            },
        )

    # ================================================================
    # Weight sensitivity analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"{'Layer':<45} | " + " | ".join(f"{b}-bit" for b in args.bit_widths))
    print("-" * (50 + 10 * len(args.bit_widths)))

    weight_results = {}
    for layer_name, layer_mod in weight_layers:
        row = {}
        accs = []
        for bits in args.bit_widths:
            acc = evaluate_single_layer_weight_quant(
                model, layer_name, bits,
                calib_dataloader, test_dataloader,
                args.n_calib_batch, device
            )
            row[bits] = acc
            accs.append(f"{acc:.4f}")

            if not args.no_mlflow:
                mlflow.log_metrics({
                    f"weight_sensitivity/{layer_name}/{bits}bit": acc,
                    f"weight_sensitivity/{layer_name}/{bits}bit_drop": baseline_acc - acc,
                })

        weight_results[layer_name] = row
        print(f"{layer_name:<45} | " + " | ".join(accs))

    # ================================================================
    # Activation sensitivity analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("ACTIVATION SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"{'Layer':<45} | " + " | ".join(f"{b}-bit" for b in args.bit_widths))
    print("-" * (50 + 10 * len(args.bit_widths)))

    act_results = {}
    for layer_name, layer_mod in act_layers:
        row = {}
        accs = []
        for bits in args.bit_widths:
            acc = evaluate_single_layer_act_quant(
                model, layer_name, bits,
                calib_dataloader, test_dataloader,
                args.n_calib_batch, device
            )
            row[bits] = acc
            accs.append(f"{acc:.4f}")

            if not args.no_mlflow:
                mlflow.log_metrics({
                    f"act_sensitivity/{layer_name}/{bits}bit": acc,
                    f"act_sensitivity/{layer_name}/{bits}bit_drop": baseline_acc - acc,
                })

        act_results[layer_name] = row
        print(f"{layer_name:<45} | " + " | ".join(accs))

    # ================================================================
    # Summary: most sensitive layers
    # ================================================================
    print("\n" + "=" * 60)
    print("MOST SENSITIVE LAYERS (highest accuracy drop at lowest bit-width)")
    print("=" * 60)

    min_bits = min(args.bit_widths)

    print("\nWeight sensitivity (sorted by accuracy drop):")
    weight_drops = [(name, baseline_acc - row[min_bits])
                    for name, row in weight_results.items()]
    weight_drops.sort(key=lambda x: x[1], reverse=True)
    for i, (name, drop) in enumerate(weight_drops[:10]):
        print(f"  {i+1}. {name}: accuracy drop = {drop:.4f} at {min_bits}-bit")

    print(f"\nActivation sensitivity (sorted by accuracy drop):")
    act_drops = [(name, baseline_acc - row[min_bits])
                 for name, row in act_results.items()]
    act_drops.sort(key=lambda x: x[1], reverse=True)
    for i, (name, drop) in enumerate(act_drops[:10]):
        print(f"  {i+1}. {name}: accuracy drop = {drop:.4f} at {min_bits}-bit")

    # Save results (JSON + detailed text report)
    import json
    results = {
        "baseline_accuracy": baseline_acc,
        "bit_widths": args.bit_widths,
        "weight_sensitivity": weight_results,
        "activation_sensitivity": act_results,
    }
    results_path = os.path.join(exp_dir, "sensitivity_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Build detailed text report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SENSITIVITY ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Experiment:        {exp_name}")
    report_lines.append(f"Baseline accuracy: {baseline_acc:.4f}")
    report_lines.append(f"Bit-widths tested: {args.bit_widths}")
    report_lines.append(f"Calibration batch: {args.n_calib_batch}")
    report_lines.append(f"Seed:              {args.seed}")
    report_lines.append("")

    # Weight sensitivity table
    report_lines.append("=" * 70)
    report_lines.append("WEIGHT SENSITIVITY (accuracy per layer at each bit-width)")
    report_lines.append("=" * 70)
    header = f"{'Layer':<45} | " + " | ".join(f"{b:>5}-bit" for b in args.bit_widths) + " | " + f"{'Drop@' + str(min_bits) + 'b':>10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for name, drop in weight_drops:
        row = weight_results[name]
        accs = " | ".join(f"{row[b]:>8.4f}" for b in args.bit_widths)
        report_lines.append(f"{name:<45} | {accs} | {drop:>10.4f}")

    report_lines.append("")

    # Activation sensitivity table
    report_lines.append("=" * 70)
    report_lines.append("ACTIVATION SENSITIVITY (accuracy per layer at each bit-width)")
    report_lines.append("=" * 70)
    header = f"{'Layer':<45} | " + " | ".join(f"{b:>5}-bit" for b in args.bit_widths) + " | " + f"{'Drop@' + str(min_bits) + 'b':>10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for name, drop in act_drops:
        row = act_results[name]
        accs = " | ".join(f"{row[b]:>8.4f}" for b in args.bit_widths)
        report_lines.append(f"{name:<45} | {accs} | {drop:>10.4f}")

    report_lines.append("")

    # Top-10 most sensitive summary
    report_lines.append("=" * 70)
    report_lines.append(f"TOP-10 MOST SENSITIVE LAYERS (by accuracy drop at {min_bits}-bit)")
    report_lines.append("=" * 70)
    report_lines.append("\nWeights:")
    for i, (name, drop) in enumerate(weight_drops[:10]):
        report_lines.append(f"  {i+1:>2}. {name:<45} drop = {drop:.4f}")
    report_lines.append("\nActivations:")
    for i, (name, drop) in enumerate(act_drops[:10]):
        report_lines.append(f"  {i+1:>2}. {name:<45} drop = {drop:.4f}")

    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    report_path = os.path.join(exp_dir, "sensitivity_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved to {report_path}")

    if not args.no_mlflow:
        mlflow.log_artifact(results_path)
        mlflow.log_artifact(report_path)
        mlflow.end_run()

    print("\nSensitivity analysis complete.")


if __name__ == "__main__":
    main()
