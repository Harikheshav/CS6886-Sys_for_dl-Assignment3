import torch
import torch.nn as nn
import torch.nn.functional as F
from MobilenetV2 import MobileNetV2
from dataloader import get_cifar10
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_print(log_file, msg):
    """Prints to console and writes to log file."""
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

if __name__ == "__main__":

    train_loader,test_loader = get_cifar10(batchsize=128)

    model = MobileNetV2(num_classes=10, dropout=0.5)
    model.to(device)

    epochs = 300
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07,momentum=0.9,weight_decay=5e-4,nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.mean().backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader, device)


        log_print("training_log.txt",f"Epoch {epoch+1:2d}: Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")
        scheduler.step()
    
    torch.save(model.state_dict(),'./checkpoints/test2_mobilenetv2_cifar10.pth')
