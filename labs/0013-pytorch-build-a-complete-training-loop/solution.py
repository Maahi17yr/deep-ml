import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.

    Returns:
        history: List of dicts, one per epoch, with keys:
            'epoch', 'train_loss', 'val_loss', 'val_accuracy'
    """
    # Keep data on the same device as the model
    device = next(model.parameters()).device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    history = []

    for epoch in range(1, epochs + 1):
        # ---------------- Training ----------------
        model.train()
        perm = torch.randperm(n_train, device=device)  # reshuffle every epoch
        running_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # Weight by batch size so the final (smaller) batch is averaged correctly
            running_loss += loss.item() * xb.shape[0]

        train_loss = running_loss / n_train

        # ---------------- Validation ----------------
        model.eval()
        val_loss_sum = 0.0
        correct = 0

        with torch.no_grad():
            # Batched so large validation sets don't blow up memory
            for start in range(0, n_val, batch_size):
                xb = X_val[start:start + batch_size]
                yb = y_val[start:start + batch_size]

                logits = model(xb)
                val_loss_sum += criterion(logits, yb).item() * xb.shape[0]
                correct += (logits.argmax(dim=1) == yb).sum().item()

        val_loss = val_loss_sum / n_val
        val_accuracy = correct / n_val

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

    return history