"""
Training script for the underwater visual odometry model.
Operation Barbarosa training pipeline.
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from combined_model import create_combined_model, DataNormalizer

def load_image_pair(prev_path, curr_path, target_size=(256, 256)):
    """
    Load and preprocess image pair
    
    Args:
        prev_path: Path to previous image
        curr_path: Path to current image
        target_size: Target image size
        
    Returns:
        Stacked image pair
    """
    # Load images
    prev_img = cv2.imread(prev_path)
    curr_img = cv2.imread(curr_path)
    
    # Resize images
    prev_img = cv2.resize(prev_img, target_size)
    curr_img = cv2.resize(curr_img, target_size)
    
    # Convert to RGB (from BGR)
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    prev_img = prev_img.astype(np.float32) / 255.0
    curr_img = curr_img.astype(np.float32) / 255.0
    
    # Stack images along channel dimension
    stacked_img = np.concatenate([prev_img, curr_img], axis=-1)
    
    return stacked_img

def data_generator(df, batch_size=8, target_size=(256, 256), normalizer=None, shuffle=True):
    """
    Generator to yield batches of data
    
    Args:
        df: DataFrame with image pairs and pose data
        batch_size: Batch size
        target_size: Target image size
        normalizer: DataNormalizer for pose normalization
        shuffle: Whether to shuffle data
        
    Yields:
        Batches of (image_pairs, pose_changes)
    """
    indices = np.arange(len(df))
    if shuffle:
        np.random.shuffle(indices)
    
    num_batches = int(np.ceil(len(df) / batch_size))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch_indices = indices[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indices), target_size[0], target_size[1], 6))
        batch_y = np.zeros((len(batch_indices), 6))
        
        for i, idx in enumerate(batch_indices):
            row = df.iloc[idx]
            
            # Get paths for previous and current images
            pair_path = row['image_pair_path']
            prev_img_path = os.path.join(pair_path, 'prev.png')
            curr_img_path = os.path.join(pair_path, 'curr.png')
            
            # Load image pair
            try:
                stacked_img = load_image_pair(prev_img_path, curr_img_path, target_size)
                batch_x[i] = stacked_img
            except Exception as e:
                print(f"Error loading images from {pair_path}: {e}")
                # Use zeros for this sample
                batch_x[i] = np.zeros((target_size[0], target_size[1], 6))
            
            # Get pose changes
            pose_changes = np.array([
                row['delta_x'],
                row['delta_y'],
                row['delta_z'],
                row['delta_yaw'],
                row['delta_roll'],
                row['delta_pitch']
            ])
            
            batch_y[i] = pose_changes
        
        # Normalize pose data if normalizer is provided
        if normalizer is not None:
            batch_y = normalizer.normalize(batch_y)
        
        yield batch_x, batch_y


def train_model(args):
    """
    Train the underwater visual odometry model
    
    Args:
        args: Command line arguments
    """
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} image pairs")
    
    # Filter by camera IDs
    train_cameras = [int(cam) for cam in args.train_cameras.split(',')]
    test_cameras = [int(cam) for cam in args.test_cameras.split(',')]
    
    print(f"Training on cameras: {train_cameras}")
    print(f"Testing on cameras: {test_cameras}")
    
    train_df = df[df['camera_id'].isin(train_cameras)].copy()
    test_df = df[df['camera_id'].isin(test_cameras)].copy()
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Further split training data into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"Final split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize data normalizer
    normalizer = DataNormalizer(method=args.normalize_method)
    
    # Extract target values for normalization
    pose_data = np.array([
        train_df['delta_x'],
        train_df['delta_y'],
        train_df['delta_z'],
        train_df['delta_yaw'],
        train_df['delta_roll'],
        train_df['delta_pitch']
    ]).T
    
    # Fit normalizer on training data
    normalizer.fit(pose_data)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save normalizer parameters
    normalizer_path = os.path.join(args.model_dir, 'normalizer.npy')
    normalizer.save(normalizer_path)
    print(f"Saved normalizer parameters to {normalizer_path}")
    
    # Create model
    input_shape = (args.image_height, args.image_width, 6)  # Stacked image pairs
    model, flownet, pose_net = create_combined_model(
        input_shape=input_shape,
        use_weighted_loss=(args.loss_type == 'weighted'),
        use_sincos_loss=(args.loss_type == 'sincos')
    )
    
    print("Model created:")
    model.summary()
    
    # Setup callbacks
    checkpoint_path = os.path.join(args.model_dir, 'model_checkpoint.h5')
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=10, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Get data generators
    train_gen = data_generator(
        train_df, 
        batch_size=args.batch_size,
        target_size=(args.image_height, args.image_width),
        normalizer=normalizer,
        shuffle=True
    )
    
    val_gen = data_generator(
        val_df, 
        batch_size=args.batch_size,
        target_size=(args.image_height, args.image_width),
        normalizer=normalizer,
        shuffle=False
    )
    
    # Calculate steps per epoch
    train_steps = int(np.ceil(len(train_df) / args.batch_size))
    val_steps = int(np.ceil(len(val_df) / args.batch_size))
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'combined_model_final.h5')
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save individual models
    flownet_path = os.path.join(args.model_dir, 'flownet_final.h5')
    pose_net_path = os.path.join(args.model_dir, 'pose_net_final.h5')
    
    flownet.save(flownet_path)
    pose_net.save(pose_net_path)
    print(f"Saved FlowNet to {flownet_path}")
    print(f"Saved Pose Network to {pose_net_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot learning rate if it was adjusted
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')
    
    history_plot_path = os.path.join(args.model_dir, 'training_history.png')
    plt.tight_layout()
    plt.savefig(history_plot_path)
    print(f"Saved training history plot to {history_plot_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_gen = data_generator(
        test_df, 
        batch_size=args.batch_size,
        target_size=(args.image_height, args.image_width),
        normalizer=normalizer,
        shuffle=False
    )
    test_steps = int(np.ceil(len(test_df) / args.batch_size))
    
    test_loss = model.evaluate(test_gen, steps=test_steps)
    print(f"Test loss: {test_loss}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'train_cameras': train_cameras,
        'test_cameras': test_cameras,
        'normalize_method': args.normalize_method,
        'loss_type': args.loss_type
    }
    
    np.save(os.path.join(args.model_dir, 'test_results.npy'), test_results)
    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train underwater visual odometry model')
    
    # Data parameters
    parser.add_argument('--data', type=str, required=True, help='Path to CSV with image pairs and trajectory data')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save model checkpoints and results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--train_cameras', type=str, required=True, help='Comma-separated list of camera IDs for training')
    parser.add_argument('--test_cameras', type=str, required=True, help='Comma-separated list of camera IDs for testing')
    
    # Model parameters
    parser.add_argument('--image_width', type=int, default=256, help='Image width for model input')
    parser.add_argument('--image_height', type=int, default=256, help='Image height for model input')
    parser.add_argument('--normalize_method', type=str, default='none', choices=['none', 'minmax', 'meanstd'],
                        help='Method for normalizing pose data')
    parser.add_argument('--loss_type', type=str, default='sincos', choices=['mse', 'weighted', 'sincos'],
                        help='Loss function type')
    
    args = parser.parse_args()
    train_model(args)