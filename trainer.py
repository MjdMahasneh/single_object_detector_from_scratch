import torch
from eval import evaluate

def train(model, train_loader, test_loader, criterion, optimizer, epochs, eval_every, device, use_scheduler, scheduler, checkpoints_dir):
    best_iou = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if use_scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)


        if (epoch+1) % eval_every == 0:
            val_iou = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val IoU: {val_iou:.4f}")

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), f"{checkpoints_dir}/best_model.pth")
                print("✅ Saved best model!")

            # set model back to train mode
            model.train()
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")