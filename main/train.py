# main/train.py
# Training MobileNet-v2 on CIFAR-10: baseline or QAT.
# Logs hyperparameters, loss, accuracy, and model size to mlflow.
#
# Usage:
#   Baseline:  python -m main.train my_exp --epochs 200 --lr 0.01 --batch_size 128
#   QAT:       python -m main.train my_exp --qat --weight_bits 8 --act_bits 8 --epochs 20 --lr 0.001

import argparse
import datetime
import json
import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from src.mobilenetv2 import mobilenet_v2
from src.quantize import (
    calibrate,
    extract_original_weights,
    freeze_all_quant,
    model_size_bytes_fp32,
    print_compression,
    swap_to_quant_modules,
)
from src.utils import (
    configure_cudnn,
    configure_mlflow,
    evaluate,
    generate_run_name,
    load_checkpoint,
    prepare_calib_dataloader,
    prepare_dataloaders,
    save_checkpoint,
    set_seed,
    test,
    train,
)


def parse_arg():
    parser = argparse.ArgumentParser(description="Train MobileNet-v2 on CIFAR-10")
    parser.add_argument("--exp_name", default='', type=str, help="Experiment name (used as model directory and mlflow run name)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_drop_epochs", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--no_mlflow", action="store_true", help="Disable mlflow logging")

    # QAT options
    parser.add_argument("--qat", action="store_true", help="Enable quantization-aware training")
    parser.add_argument("--weight_bits", type=int, default=8, help="Bit-width for weight quantization (QAT)")
    parser.add_argument("--act_bits", type=int, default=8, help="Bit-width for activation quantization (QAT)")
    parser.add_argument("--skip_layers", nargs="*", default=[],
                        help="Layer name prefixes to skip quantization on (QAT)")
    parser.add_argument("--n_calib_batch", type=int, default=32,
                        help="Number of batches for initial activation calibration (QAT)")
    return parser.parse_args()


def main():
    args = parse_arg()

    set_seed(args.seed)
    configure_cudnn(deterministic=True, benchmark=False)

    exp_name = args.exp_name
    exp_dir = os.path.join(args.model_dir, exp_name)
    mode = "qat" if args.qat else "baseline"

    # Prepare directory
    if args.resume:
        assert os.path.exists(
            os.path.join(exp_dir, "checkpoint_latest.pth")
        ), "Cannot find checkpoint file for resuming."
    else:
        os.makedirs(exp_dir, exist_ok=True)

    # Dump config
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(exp_dir, f"config_{mode}_{time_str}.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Mode: {mode}")

    # Prepare data
    print("Preparing dataset...")
    train_dataloader, test_dataloader = prepare_dataloaders(args.batch_size)

    # Prepare model
    print("Preparing model...")
    model = mobilenet_v2(num_classes=args.num_classes)

    # For QAT: load pre-trained baseline first
    baseline_acc = None
    if args.qat:
        model_path = os.path.join(exp_dir, "best_model.pth")
        assert os.path.exists(model_path), \
            f"QAT requires a pre-trained baseline. Train baseline first. Missing: {model_path}"
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded pre-trained baseline from {model_path}")

        # Evaluate baseline before QAT
        model.to(device)
        baseline_acc = test(model, device, test_dataloader)
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        model.cpu()

        # Swap to quantized modules with STE
        skip_set = set(args.skip_layers)
        print(f"\nPreparing QAT: weight_bits={args.weight_bits}, act_bits={args.act_bits}")
        if skip_set:
            print(f"  Skipping layers: {skip_set}")
        swap_to_quant_modules(
            model, weight_bits=args.weight_bits, act_bits=args.act_bits,
            qat=True, skip_layers=skip_set
        )
        model.to(device)

        # Calibrate activation ranges before QAT starts
        print("Initial calibration of activation ranges...")
        calib_dataloader = prepare_calib_dataloader(args.batch_size)
        calibrate(model, calib_dataloader, args.n_calib_batch, device=device)
        freeze_all_quant(model)

    else:
        model.to(device)

    # Log model size
    fp32_size = model_size_bytes_fp32(model)
    print(f"FP32 model size: {fp32_size / 1024 / 1024:.2f} MB")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_drop_epochs, gamma=0.1
    )
    start_epoch = 0
    best_accuracy = -1
    epoch_at_best_accuracy = -1

    if args.resume and not args.qat:
        model, optimizer, scheduler, start_epoch, best_accuracy, epoch_at_best_accuracy = load_checkpoint(
            os.path.join(exp_dir, "checkpoint_latest.pth"),
            model, optimizer, scheduler, start_epoch, best_accuracy, epoch_at_best_accuracy,
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best accuracy so far: {best_accuracy:.4f}")

    # Initialize mlflow
    run_name = exp_name or generate_run_name("train", mode, args.epochs, args.batch_size, args.lr)
    mlflow_config = {
        "exp_name": exp_name,
        "mode": mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_drop_epochs": str(args.lr_drop_epochs),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "fp32_model_size_mb": fp32_size / 1024 / 1024,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    if args.qat:
        mlflow_config.update({
            "weight_bits": args.weight_bits,
            "act_bits": args.act_bits,
            "skip_layers": str(args.skip_layers),
            "baseline_accuracy": baseline_acc,
        })

    if not args.no_mlflow:
        configure_mlflow(
            experiment_name="cs6886_mobilenetv2_quantization",
            run_name=run_name,
            config=mlflow_config,
        )

    # Training loop
    print(f"\nStarting {mode} training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch: {epoch}/{args.epochs}, lr: {lr:.6f}")

        # Train
        loss_epoch, train_accuracy = train(model, optimizer, scheduler, criterion, device, train_dataloader)
        print(f"  Train loss: {loss_epoch:.6f}")
        print(f"  Train accuracy: {train_accuracy:.4f}")

        # Evaluate
        test_loss, accuracy = evaluate(model, device, test_dataloader, criterion=criterion)
        print(f"  Test loss: {test_loss:.6f}")
        print(f"  Test accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epoch_at_best_accuracy = epoch
            print(f"  New best accuracy! Saving model...")

            if args.qat:
                # Save clean FP32 weights (compatible with vanilla model loading)
                save_path = os.path.join(exp_dir, "best_model_qat.pth")
                clean_sd = extract_original_weights(model)
                torch.save(clean_sd, save_path)
            else:
                save_path = os.path.join(exp_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)

        # Save checkpoint (baseline only, QAT is short enough to not need it)
        if not args.qat:
            save_checkpoint(
                os.path.join(exp_dir, "checkpoint_latest.pth"),
                model, optimizer, scheduler, epoch, best_accuracy, epoch_at_best_accuracy,
            )

        # Log to mlflow
        if not args.no_mlflow:
            mlflow.log_metrics({
                "lr": lr,
                "train/loss": loss_epoch,
                "train/accuracy": train_accuracy,
                "test/loss": test_loss,
                "test/accuracy": accuracy,
                "test/best_accuracy": best_accuracy,
            }, step=epoch)

    print(f"\nTraining complete. Best accuracy: {best_accuracy:.4f} at epoch {epoch_at_best_accuracy}")
    if baseline_acc is not None:
        print(f"Baseline accuracy was: {baseline_acc:.4f}")

    # Final logging
    if not args.no_mlflow:
        final_metrics = {
            "final/best_accuracy": best_accuracy,
            "final/epoch_at_best_accuracy": epoch_at_best_accuracy,
            "final/fp32_model_size_mb": fp32_size / 1024 / 1024,
        }
        if args.qat:
            example_input = torch.randn(1, 3, 32, 32, device=device)
            comp_stats = print_compression(
                model, weight_bits=args.weight_bits, act_bits=args.act_bits,
                input_tensor=example_input
            )
            final_metrics.update({
                "final/baseline_accuracy": baseline_acc,
                "final/weight_bits": args.weight_bits,
                "final/act_bits": args.act_bits,
                "final/quant_model_size_mb": comp_stats["quant_model_size_mb"],
                "final/weight_compression_ratio": comp_stats["weight_compression_ratio"],
            })
        mlflow.log_metrics(final_metrics)
        mlflow.end_run()


if __name__ == "__main__":
    main()
