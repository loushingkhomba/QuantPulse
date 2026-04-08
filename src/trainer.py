import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=5000,
    batch_size=64,
    save_every=100,
    model_path=None,
    patience=20,
    checkpoint_path=None,
    lr=3e-4,
    weight_decay=5e-4,
    grad_clip=1.0,
    label_smoothing=0.05,
    confidence_penalty=0.0
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_counts = torch.bincount(y_train.detach().cpu(), minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * torch.clamp(class_counts, min=1.0))
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    print(
        "Class weights:",
        [round(float(class_weights[0].item()), 4), round(float(class_weights[1].item()), 4)]
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    last_train_loss = None
    last_val_loss = None

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            # Penalize overconfident logits to reduce noisy overtrading downstream.
            if confidence_penalty > 0:
                probs = torch.softmax(outputs, dim=1)
                p1 = probs[:, 1]
                smooth_penalty = torch.mean((p1 - 0.5) ** 2)
                loss = loss + (confidence_penalty * smooth_penalty)

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --------------------
        # Validation
        # --------------------

        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        last_train_loss = avg_train_loss
        last_val_loss = avg_val_loss

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch} | Train Loss {avg_train_loss:.5f} | Val Loss {avg_val_loss:.5f}"
        )

        # --------------------
        # Early Stopping
        # --------------------

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            if model_path:
                torch.save(model.state_dict(), model_path)
                print("Best model saved.")

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("Early stopping triggered.")
            break

        # periodic checkpoint
        if checkpoint_path and epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_path)
            print("Checkpoint saved.")

    return {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "last_train_loss": float(last_train_loss) if last_train_loss is not None else None,
        "last_val_loss": float(last_val_loss) if last_val_loss is not None else None,
        "class_weight_0": float(class_weights[0].item()),
        "class_weight_1": float(class_weights[1].item())
    }