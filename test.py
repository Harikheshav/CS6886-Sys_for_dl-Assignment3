import torch
import torch.nn as nn
from MobilenetV2 import MobileNetV2
from dataloader import get_cifar10
from utils import *
from quantize import *
import argparse

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
    print(f"Test Acc={test_acc:.2f}%")

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
    print_compression(model,weight_bits=weight_quantize_bits)
