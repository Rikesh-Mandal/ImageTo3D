
import os
import torch

from tqdm import tqdm

from src.data_loader import data_generators
from src.model import (
    encoder_resnet,
    decoder,
    loss_function,
    create_optimizer,
    save_model,
    load_model
)
from src.train import train_step, validate
from src.eval import evaluate
from src.visualization import plot_results


def main():
    # Choose GPU when available, otherwise CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Original dissertation dataset location.
    # Update this only if training is required again.
    data_root = (
        "E:/Dissertation/Trial/ImageTo3D/"
        "ImageTo3D/data/NYU_Depth_V2"
    )

    # Store checkpoints relative to the project directory.
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    save_dir = os.path.join(
        project_root,
        "saved_models"
    )

    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(
        save_dir,
        "best_model.pth"
    )

    batch_size = 10
    epochs = 21

    print("Creating data generators...")

    (
        train_data,
        val_data,
        test_data,
        train_len,
        val_len,
        test_len
    ) = data_generators(data_root, batch_size)

    print(f"Train Data Length: {train_len}")
    print(f"Validation Data Length: {val_len}")
    print(f"Test Data Length: {test_len}")

    train_steps_per_epoch = train_len // batch_size
    val_steps = val_len // batch_size
    test_steps = test_len // batch_size

    print("Initializing model...")

    backbone = encoder_resnet().to(device)
    head = decoder().to(device)

    loss_fn = loss_function()

    optimizer = create_optimizer(
        backbone,
        head
    )

    print("\nVerifying model dimensions...")

    shape_test_rgb, shape_test_depth = next(train_data)

    print(
        f"Input RGB shape: {shape_test_rgb.shape} "
        "(expected [batch, 3, 384, 384])"
    )

    print(
        f"Target depth shape: {shape_test_depth.shape} "
        "(expected [batch, 1, 384, 384])"
    )

    # Dry run before training.
    with torch.no_grad():

        test_features = backbone(
            shape_test_rgb[:1].float().to(device)
        )

        print(
            f"Backbone output shape: {test_features.shape} "
            "(expected [1, 2048, 12, 12])"
        )

        test_pred = head(test_features)

        print(
            f"Decoder output shape: {test_pred.shape} "
            "(expected [1, 1, 384, 384])"
        )

    best_val_loss = float("inf")

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        backbone.train()
        head.train()

        train_loss = 0.0

        for _ in tqdm(
            range(train_steps_per_epoch),
            desc="Training"
        ):

            batch = next(train_data)

            loss = train_step(
                backbone,
                head,
                batch,
                loss_fn,
                optimizer,
                device
            )

            train_loss += loss

        avg_train_loss = (
            train_loss / train_steps_per_epoch
        )

        val_loss = validate(
            backbone,
            head,
            val_data(),
            loss_fn,
            device,
            val_steps
        )

        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_model(
                backbone,
                head,
                model_path
            )

            print("Saved new best model")

    print("\nLoading best model for evaluation...")

    backbone, head = load_model(
        model_path,
        device
    )

    metrics = evaluate(
        backbone,
        head,
        test_data(),
        device,
        test_steps
    )

    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Absolute Relative Error: {metrics['abs_rel']:.4f}")

    print("\nGenerating sample prediction...")

    rgb_sample, depth_sample = next(test_data())

    with torch.no_grad():

        features = backbone(
            rgb_sample[:1].float().to(device)
        )

        pred_sample = head(features)

    if pred_sample.shape[-2:] != depth_sample[:1].shape[-2:]:

        pred_sample = torch.nn.functional.interpolate(
            pred_sample,
            size=depth_sample[:1].shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    prediction_path = os.path.join(
        save_dir,
        "sample_prediction.png"
    )

    plot_results(
        rgb_sample[0],
        depth_sample[0],
        pred_sample[0],
        prediction_path
    )


if __name__ == "__main__":
    main()
