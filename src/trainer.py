import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _pairwise_rank_loss(scores, labels, dates):
    unique_dates = torch.unique(dates)
    per_date_losses = []
    for date_value in unique_dates:
        mask = dates == date_value
        if torch.sum(mask) < 2:
            continue
        day_scores = scores[mask]
        day_labels = labels[mask]

        pos_scores = day_scores[day_labels > 0.5]
        neg_scores = day_scores[day_labels <= 0.5]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue

        # Logistic pairwise ranking: maximize score(pos) - score(neg).
        diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        per_date_losses.append(torch.nn.functional.softplus(-diffs).mean())

    if not per_date_losses:
        return None
    return torch.stack(per_date_losses).mean()


def _build_regime_weights(regime_tensor, stress_weight, neutral_weight, bullish_weight):
    regime_tensor = regime_tensor.to(dtype=torch.float32)
    weights = torch.full_like(regime_tensor, fill_value=neutral_weight)
    weights = torch.where(regime_tensor < 0, torch.full_like(weights, stress_weight), weights)
    weights = torch.where(regime_tensor > 0, torch.full_like(weights, bullish_weight), weights)
    return weights


def _false_positive_cost(outputs, labels, power=2.0, threshold=0.55, gate_sharpness=20.0):
    """Penalize confident positive predictions on negative labels.

    Uses a smooth gate around `threshold` so only predicted-alpha negatives get a heavy cost.
    """
    probs = torch.softmax(outputs, dim=1)[:, 1]
    negative_mask = (labels == 0).to(dtype=probs.dtype)
    negative_count = torch.clamp(negative_mask.sum(), min=1.0)
    pred_positive_gate = torch.sigmoid((probs - threshold) * gate_sharpness)
    fp_excess = torch.clamp(probs - threshold, min=0.0)
    fp_pressure = pred_positive_gate * torch.pow(fp_excess, power)
    return torch.sum(fp_pressure * negative_mask) / negative_count


def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    regime_train=None,
    regime_val=None,
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
    confidence_penalty=0.0,
    regime_robust_training=True,
    regime_stress_weight=1.35,
    regime_bullish_weight=0.90,
    objective_mode="classification",
    rank_loss_weight=1.0,
    classification_loss_weight=0.25,
    train_dates=None,
    val_dates=None,
    false_positive_cost_multiplier=0.0,
    false_positive_cost_power=2.0,
    false_positive_cost_threshold=0.55,
    false_positive_cost_gate_sharpness=20.0,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    include_dates = objective_mode == "ranking" and train_dates is not None and val_dates is not None

    if include_dates:
        if regime_train is not None:
            train_dataset = TensorDataset(X_train, y_train, regime_train, train_dates)
        else:
            train_dataset = TensorDataset(X_train, y_train, train_dates)
        if regime_val is not None:
            val_dataset = TensorDataset(X_val, y_val, regime_val, val_dates)
        else:
            val_dataset = TensorDataset(X_val, y_val, val_dates)
    else:
        if regime_train is not None:
            train_dataset = TensorDataset(X_train, y_train, regime_train)
        else:
            train_dataset = TensorDataset(X_train, y_train)

        if regime_val is not None:
            val_dataset = TensorDataset(X_val, y_val, regime_val)
        else:
            val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(objective_mode != "ranking"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_counts = torch.bincount(y_train.detach().cpu(), minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * torch.clamp(class_counts, min=1.0))
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing, reduction="none")

    print(
        "Class weights:",
        [round(float(class_weights[0].item()), 4), round(float(class_weights[1].item()), 4)]
    )
    print(
        "Transaction-aware FP cost:",
        {
            "multiplier": float(false_positive_cost_multiplier),
            "power": float(false_positive_cost_power),
            "threshold": float(false_positive_cost_threshold),
            "gate_sharpness": float(false_positive_cost_gate_sharpness),
        },
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

        for batch in train_loader:

            if include_dates and regime_robust_training and len(batch) == 4:
                batch_X, batch_y, batch_regime, batch_dates = batch
            elif include_dates and len(batch) == 3:
                batch_X, batch_y, batch_dates = batch
                batch_regime = None
            elif regime_robust_training and len(batch) == 3:
                batch_X, batch_y, batch_regime = batch
                batch_dates = None
            else:
                batch_X, batch_y = batch
                batch_regime = None
                batch_dates = None

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            if batch_regime is not None:
                batch_regime = batch_regime.to(device)
            if batch_dates is not None:
                batch_dates = batch_dates.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            sample_loss = criterion(outputs, batch_y)

            if regime_robust_training and batch_regime is not None:
                regime_weights = _build_regime_weights(
                    batch_regime,
                    stress_weight=regime_stress_weight,
                    neutral_weight=1.0,
                    bullish_weight=regime_bullish_weight,
                )
                sample_loss = sample_loss * regime_weights

            cls_loss = sample_loss.mean()
            if objective_mode == "ranking" and batch_dates is not None:
                rank_scores = outputs[:, 1] - outputs[:, 0]
                rank_loss = _pairwise_rank_loss(rank_scores, batch_y.float(), batch_dates)
                if rank_loss is not None:
                    loss = (classification_loss_weight * cls_loss) + (rank_loss_weight * rank_loss)
                else:
                    loss = cls_loss
            else:
                loss = cls_loss

            # Transaction-aware objective: heavily penalize false-positive alpha calls.
            if false_positive_cost_multiplier > 0:
                fp_cost = _false_positive_cost(
                    outputs,
                    batch_y,
                    power=false_positive_cost_power,
                    threshold=false_positive_cost_threshold,
                    gate_sharpness=false_positive_cost_gate_sharpness,
                )
                loss = loss + (false_positive_cost_multiplier * fp_cost)

            # Penalize overconfident logits to reduce noisy overtrading downstream.
            if confidence_penalty > 0:
                probs = torch.softmax(outputs, dim=1)
                p1 = probs[:, 1]
                smooth_penalty = (p1 - 0.5) ** 2
                if regime_robust_training and batch_regime is not None:
                    confidence_weights = _build_regime_weights(
                        batch_regime,
                        stress_weight=1.50,
                        neutral_weight=1.0,
                        bullish_weight=0.85,
                    )
                    smooth_penalty = smooth_penalty * confidence_weights
                smooth_penalty = smooth_penalty.mean()
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

            for batch in val_loader:

                if include_dates and len(batch) == 4:
                    batch_X, batch_y, _, batch_dates = batch
                elif include_dates and len(batch) == 3:
                    batch_X, batch_y, batch_dates = batch
                elif len(batch) == 3:
                    batch_X, batch_y, _ = batch
                    batch_dates = None
                else:
                    batch_X, batch_y = batch
                    batch_dates = None

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                if batch_dates is not None:
                    batch_dates = batch_dates.to(device)

                outputs = model(batch_X)

                val_cls_loss = criterion(outputs, batch_y).mean()
                if objective_mode == "ranking" and batch_dates is not None:
                    val_scores = outputs[:, 1] - outputs[:, 0]
                    val_rank_loss = _pairwise_rank_loss(val_scores, batch_y.float(), batch_dates)
                    if val_rank_loss is not None:
                        loss = (classification_loss_weight * val_cls_loss) + (rank_loss_weight * val_rank_loss)
                    else:
                        loss = val_cls_loss
                else:
                    loss = val_cls_loss

                if false_positive_cost_multiplier > 0:
                    fp_cost = _false_positive_cost(
                        outputs,
                        batch_y,
                        power=false_positive_cost_power,
                        threshold=false_positive_cost_threshold,
                        gate_sharpness=false_positive_cost_gate_sharpness,
                    )
                    loss = loss + (false_positive_cost_multiplier * fp_cost)

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