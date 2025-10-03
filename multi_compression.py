import torch
import torch.nn as nn
from MobilenetV2 import MobileNetV2
from dataloader import get_cifar10
from utils import *
from quantize import *
import argparse
import matplotlib.pyplot as plt
import os
from pandas.plotting import parallel_coordinates
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def approximate_compression_ratio(num_params, orig_bits, quant_bits):
    orig_size_bits = num_params * orig_bits
    quant_size_bits = num_params * quant_bits
    return orig_size_bits / quant_size_bits if quant_size_bits > 0 else float('inf')

def calibrate_model(model, loader, device, num_batches=5):
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            _ = model(x)
            if i + 1 >= num_batches:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantization sweep')
    parser.add_argument('--weight_bits', type=str, default=None,
                        help='Comma-separated list of weight bits to try, e.g. "8,6,4".')
    parser.add_argument('--act_bits', type=str, default=None,
                        help='Comma-separated list of activation bits to try, e.g. "8,6,4".')
    parser.add_argument('--calib_batches', type=int, default=5, help='Number of batches to use for calibration.')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size for dataloaders.')
    parser.add_argument('--weights_path', type=str, default='./checkpoints/test2_mobilenetv2_cifar10.pth',
                        help='Path to FP32 model weights.')
    args = parser.parse_args()

    weight_bits_list = [int(x) for x in args.weight_bits.split(',')] if args.weight_bits else [8, 6, 4, 2]
    act_bits_list = [int(x) for x in args.act_bits.split(',')] if args.act_bits else [8, 6, 4]


    train_loader, test_loader = get_cifar10(batchsize=args.batchsize)

    # FP32 baseline
    base_model = MobileNetV2(num_classes=10, dropout=0.5)
    try:
        base_model.load_state_dict(torch.load(args.weights_path, weights_only=True, map_location=device))
    except TypeError:
        base_model.load_state_dict(torch.load(args.weights_path, map_location=device))
    base_model.to(device).eval()
    baseline_test_acc = evaluate(base_model, test_loader, device)
    total_params = count_parameters(base_model)
    print(f"FP32 baseline Test Acc={baseline_test_acc:.2f}%, total params={total_params:,}")

    results = []

    for wbits in weight_bits_list:
        for abits in act_bits_list:
            print(f"\n--- Quantization: weight_bits={wbits}, act_bits={abits} ---")
            model = MobileNetV2(num_classes=10, dropout=0.5)
            try:
                model.load_state_dict(torch.load(args.weights_path, weights_only=True, map_location=device))
            except TypeError:
                model.load_state_dict(torch.load(args.weights_path, map_location=device))
            model.to(device).eval()

            swap_to_quant_modules(model, weight_bits=wbits, act_bits=abits, activations_unsigned=True)
            calibrate_model(model, train_loader, device, num_batches=args.calib_batches)
            freeze_all_quant(model)

            with torch.no_grad():
                test_acc = evaluate(model, test_loader, device)

            comp_ratio = approximate_compression_ratio(total_params, 32, wbits)

            try:
                print_compression(model, weight_bits=wbits)
            except Exception as e:
                print(f"print_compression failed: {e}")

            print(f"Quantized Test Acc (w{wbits}/a{abits}) = {test_acc:.2f}%, compression_ratio_vs_fp32 ≈ {comp_ratio:.2f}x")
            results.append({'wbits': wbits, 'abits': abits, 'test_acc': test_acc, 'comp_ratio': comp_ratio})

            del model
            torch.cuda.empty_cache()

# Convert results to DataFrame
df = pd.DataFrame([
    {
        'ActBits': r['abits'],
        'WeightBits': r['wbits'],
        'CompressionRatio': r['comp_ratio'],
        'ModelSizeMB': (total_params * r['wbits']) / (8*1024**2),
        'QuantAcc': r['test_acc']
    }
    for r in results
])

# Normalize values for parallel plot
cols = ['ActBits','WeightBits','CompressionRatio','ModelSizeMB','QuantAcc']
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

# Plot
fig, ax = plt.subplots(figsize=(12,6))
cmap = plt.cm.viridis
norm = plt.Normalize(df['QuantAcc'].min(), df['QuantAcc'].max())

for i in range(len(df_scaled)):
    values = df_scaled.iloc[i].values
    color = cmap(norm(df['QuantAcc'].iloc[i]))  # color by accuracy
    ax.plot(cols, values, marker='o', color=color, alpha=0.8)

# Colorbar for accuracy (attach explicitly to ax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required for matplotlib < 3.6
fig.colorbar(sm, ax=ax, label="Quantized Accuracy (%)")

ax.set_title("Parallel Coordinates Plot: Quantization Metrics (colored by accuracy)")
ax.grid(True, alpha=0.3)
ax.set_ylabel("Normalized Scale [0-1]")
plt.tight_layout()

pc_plot_path = "parallel_coordinates.png"
plt.savefig(pc_plot_path, dpi=300)
print(f"Saved Parallel Coordinates plot: {pc_plot_path}")
