import torch
import torch.nn as nn
import torch.optim as optim
from mobilenet import *
from dataloader import *
from utils import *

# Script to continue training from last trained weights

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(device_name)

# Example: your model
model = MobileNetV2(num_classes=10).to(device)
num_epochs = 50

# ---- 1. Load checkpoint ----
checkpoint = torch.load("./checkpoints/opt_mobilenetv2_cifar10.pth", map_location=device_name)
model.load_state_dict(checkpoint)

train_loader, test_loader = get_cifar10(batchsize=256)

# ---- 2. Recreate optimizer and load state ----
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ---- 3. Restore epoch (optional) ----
loss_fn = nn.CrossEntropyLoss()

# ---- 4. Continue training ----
# Training Loop
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients (Reset the gradients to prevent new gradients from being added with old ones)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
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
    
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
    

test_acc = evaluate(model, test_loader, device)
    
print(f"                                | Test Acc: {test_acc:.2f}%")
print("--- Training finished successfully ---")
    

torch.save(model.state_dict(), './checkpoints/opt2_mobilenetv2_cifar10.pth')
print("Model saved to opt_mobilenetv2_cifar10.pth")

