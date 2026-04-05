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
    patience=20
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

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

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch} | Train Loss {avg_train_loss:.5f} | Val Loss {avg_val_loss:.5f}"
        )

        # --------------------
        # Early Stopping
        # --------------------

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
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
        if model_path and epoch % save_every == 0:
            torch.save(model.state_dict(), model_path)
            print("Checkpoint saved.")