import torch
import torch.nn as nn
from MobilenetV2 import MobileNetV2
from dataloader import get_cifar10
from utils import *
from quantize import *
import argparse
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantization')
    parser.add_argument('--weight_quant_bits',type=int,default=8,help='Bits to Quantize the weights')
    parser.add_argument('--activation_quant_bits',type=int,default=8,help='Activation quantization bits')

    args = parser.parse_args()
    train_loader,test_loader = get_cifar10(batchsize=2)
    model = MobileNetV2(num_classes=10, dropout=0.5)
    model.load_state_dict(torch.load('./checkpoints/test2_mobilenetv2_cifar10.pth',weights_only=True,map_location = device))
    model.to(device)
    model.eval()
    test_acc = evaluate(model, test_loader, device)
    epochs = parse_training_log("training_log.txt")["epochs"]
    test_accs = parse_training_log("training_log.txt")["test_accs"]
    train_accs = parse_training_log("training_log.txt")["train_accs"]
    train_losses = parse_training_log("training_log.txt")["train_losses"]
    print(f"Test Acc={test_acc:.2f}%")
    # Plotting
    plt.figure(figsize=(12,5))

    # Loss curve
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Train Acc", color="green")
    plt.plot(epochs, test_accs, label="Test Acc", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # performing Quantization
    weight_quantize_bits = args.weight_quant_bits
    act_quantize_bits = args.activation_quant_bits

    swap_to_quant_modules(model, weight_bits=weight_quantize_bits, act_bits=act_quantize_bits, activations_unsigned=True)
    model.to(device)
    with torch.no_grad():
        for i,(x,_) in enumerate(train_loader):
            x = x.to(device)
            y = model(x)
            if i>=100: # calibration with 5 batches
                break

    freeze_all_quant(model)
    quantize_test_acc = evaluate(model, test_loader, device)
    print(f"Quantized Test Acc={quantize_test_acc:.2f}%")
    print_compression(model,
                  weight_bits=weight_quantize_bits,
                  act_bits=act_quantize_bits,
                  device=device)
