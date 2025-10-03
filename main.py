import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import get_cifar10
from utils import *
from mobilenet import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_mobilenet_v2_cifar10(epochs=10, batch_size=128, learning_rate=0.01):
    print("--- Starting MobileNetV2 CIFAR-10 Training ---")
    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    # Data Loaders
    train_loader, test_loader = get_cifar10(batch_size)
    
    # Model Initialization (CIFAR-10 has 10 classes)
    net = MobileNetV2(num_classes=10).to(device)
    
    # Loss Function and Optimizer (Standard Loss capture criteria)
    criterion = nn.CrossEntropyLoss()
    
    # SDG + momentum optimizer works well for small dataset like CIFAR-10 (The original Mobilenet-v2 paper uses RMSprop)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler to reduce LR (Learning Rate), a common practice
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Training Loop
    net.train()
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients (Reset the gradients to prevent new gradients from being added with old ones)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics generation
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step() # Update learning rate (Dynamic LR scheduler)

        train_acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        
        # train_losses.append(train_loss / len(train_loader))
        # train_accs.append(running_acc / len(train_loader))
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Optional: Run evaluation after each epoch
        test_acc = evaluate(net, test_loader, criterion, device)
        print(f"                                | Test Acc: {test_acc:.2f}%")

    print("--- Training finished successfully ---")
    
    # Save the final model
    torch.save(net.state_dict(), './checkpoints/opt_mobilenetv2_cifar10.pth')
    print("Model saved to mobilenetv2_cifar10.pth")
    


if __name__ == '__main__':
    # Adjust hyperparameters as needed
    train_mobilenet_v2_cifar10(epochs=250, batch_size=128, learning_rate=0.1)