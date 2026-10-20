import os
import torch
from tqdm import tqdm
from data_loader import data_generators
from model import encoder_resnet, decoder, loss_function, create_optimizer, save_model, load_model
from train import train_step, validate
from eval import evaluate
from visualization import plot_results
from live_plot import live_plot, update_live_plot


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    data_root = "E:/Dissertation/Trial/ImageTo3D/ImageTo3D/data/NYU_Depth_V2"
    save_dir = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pth")

    batch_size = 10
    epochs = 21

    # Initialize components
    print("Creating data generators...")
    train_data, val_data, test_data, train_len, val_len, test_len = data_generators(data_root, batch_size)
    print(f"Train Data Length: {train_len}\n")
    print(f"Validation Data Length: {val_len}\n")
    print(f"Test Data Length: {test_len}\n")

    train_steps_per_epoch = train_len // batch_size
    val_steps = val_len // batch_size
    test_steps = test_len // batch_size

    print("Initializing model...")
    '''
    calling the function everytime when needed would create a 
    new model which resets weights, waste memory and the training
    doesnt accumulate over time
    hence initialization is required
    '''
    backbone = encoder_resnet().to(device)
    head = decoder().to(device)
    loss_fn = loss_function()
    optimizer = create_optimizer(backbone, head)

    # Verify dimensions before training
    print("\nVerifying dimensions:")
    shape_test_rgb, shape_test_depth = next(train_data)  # assigns the next batch to the variables
    print(f"Input RGB shape: {shape_test_rgb.shape} (should be [batch, 3, 384, 384])")
    print(f"Target depth shape: {shape_test_depth.shape} (should be [batch, 1, 384, 384])")

    '''
    Dry-Run
    testing if everything connects properly before
    committing to training
    '''
    with torch.no_grad():  # not tracking gradients
        test_features = backbone(shape_test_rgb[:1].float().to(device))  # taking one sample of input rgb
        print(f"Backbone output shape: {test_features.shape} (should be [1, 2048, 12, 12])")
        test_pred = head(test_features)
        print(f"Head output shape: {test_pred.shape} (should be [1, 1, 384, 384])")

    # Training loop
    '''
     any upcoming loss will be less than this and the model with best val loss will be saved
    '''
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")  # currently running epoch

        # Training
        backbone.train()
        head.train()
        train_loss = 0.0

        for _ in tqdm(range(train_steps_per_epoch), desc="Training"):
            batch = next(train_data)
            loss = train_step(backbone, head, batch, loss_fn, optimizer, device)
            train_loss += loss

        avg_train_loss = train_loss / train_steps_per_epoch  # avg loss across all batches for current epoch

        # Validation
        val_loss = validate(backbone, head, val_data(), loss_fn, device, val_steps)  # avg loss on unseen data

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(backbone, head, model_path)
            print("Saved new best model")

    # Load the best model before final evaluation and prediction
    print("\nLoading best model for final evaluation...")
    backbone, head = load_model(model_path, device)

    # Final evaluation on test data
    print("\nRunning final evaluation...")
    metrics = evaluate(backbone, head, test_data(), device, test_steps)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Absolute Relative Error: {metrics['abs_rel']:.4f}")

    # Visualize sample prediction
    print("\nGenerating sample prediction...")
    rgb_sample, depth_sample = next(test_data())  # 1 batch of 10 images
    with torch.no_grad():
        features = backbone(rgb_sample[:1].float().to(device))  # slicing the first image in the batch
        pred_sample = head(features)

    # Ensure visualization matches dimensions
    '''
    comparing if the shape of height & width of predicted sample
    matches the depth_sample's height $ width shape
    '''
    if pred_sample.shape[-2:] != depth_sample[:1].shape[-2:]:
        pred_sample = torch.nn.functional.interpolate(
            pred_sample,
            size=depth_sample[:1].shape[-2:],
            mode='bilinear'
        )

    sample_save_dir = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\saved_models"
    prediction_path = os.path.join(sample_save_dir, "sample_prediction.png")
    plot_results(rgb_sample[0], depth_sample[0], pred_sample[0], prediction_path)


if __name__ == "__main__":
    main()