import torch
import torch.nn as nn
import torch.nn.functional as F


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

# @TODO: Understand this
def distillation_loss(student_logits, labels, teacher_logits, temp, alpha):
    """
    Calculates the distillation loss.
    """
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1)
    ) * (temp * temp)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1. - alpha) * hard_loss


# @TODO: Understand this
def fine_tune_with_distillation(teacher_model, student_model, loader, epochs=3):
    """
    A simple fine-tuning loop using knowledge distillation.
    """
    print("\n--- Fine-tuning with Knowledge Distillation ---")

    
    # Teacher model is frozen
    teacher_model.eval()
    
    # Student model is in training mode
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    print(f"Starting fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Get teacher predictions (no gradients needed)
            with torch.no_grad():
                # print("Inputs: ", inputs)
                teacher_logits = teacher_model(inputs)
            
            # Get student predictions
            student_logits = student_model(inputs)
            
            # Calculate loss
            loss = distillation_loss(
                student_logits=student_logits,
                labels=labels,
                teacher_logits=teacher_logits,
                temp=2.0,  # Temperature softens probability distributions
                alpha=0.5  # Weight between soft and hard loss
            )
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 500 == 499: # Print every 50 mini-batches (dummy loader only has 4)
                 print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 50:.3f}')
                 running_loss = 0.0

    print("Finished fine-tuning.")
    student_model.eval()
    return student_model