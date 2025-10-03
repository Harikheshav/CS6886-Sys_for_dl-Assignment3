import torch 
import torch.nn as nn
import re 

def evaluate(model, testloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
    return final_acc

def parse_training_log(log_file):
    epochs = []
    train_losses = []
    train_accs = []
    test_accs = []

    pattern = r"Epoch\s+(\d+): Loss=([\d\.]+) \| Train Acc=([\d\.]+)% \| Test Acc=([\d\.]+)%"

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))
                test_accs.append(float(match.group(4)))

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs
    }