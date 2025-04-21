"""
Utility functions for data processing in Operation Barbarosa.
Handles image loading, preprocessing, and data augmentation.
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def preprocess_image(img, target_size=(256, 256)):
    """
    Preprocess an image for model input
    
    Args:
        img: Input image (BGR format from cv2.imread)
        target_size: Target image size (width, height)
        
    Returns:
        Preprocessed image
    """
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def create_dataset_from_csv(csv_path, target_size=(256, 256), batch_size=8, shuffle=True, 
                           normalizer=None, augment=False):
    """
    Create a TensorFlow dataset from a CSV file
    
    Args:
        csv_path: Path to CSV file with image pairs and pose data
        target_size: Target image size
        batch_size: Batch size
        shuffle: Whether to shuffle data
        normalizer: Normalizer for pose data
        augment: Whether to apply data augmentation
        
    Returns:
        TensorFlow dataset
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Create lists for image pairs and pose changes
    image_paths = []
    pose_changes = []
    
    for _, row in df.iterrows():
        pair_path = row['image_pair_path']
        prev_img_path = os.path.join(pair_path, 'prev.png')
        curr_img_path = os.path.join(pair_path, 'curr.png')
        
        # Check if image files exist
        if os.path.exists(prev_img_path) and os.path.exists(curr_img_path):
            image_paths.append((prev_img_path, curr_img_path))
            
            # Extract pose changes
            pose = np.array([
                row['delta_x'],
                row['delta_y'],
                row['delta_z'],
                row['delta_yaw'],
                row['delta_roll'],
                row['delta_pitch']
            ])
            
            pose_changes.append(pose)
    
    # Convert to numpy arrays
    pose_changes = np.array(pose_changes)
    
    # Normalize pose data if normalizer is provided
    if normalizer is not None:
        pose_changes = normalizer.normalize(pose_changes)
    
    # Create a dataset loading function
    def load_and_preprocess(idx):
        prev_path, curr_path = image_paths[idx]
        
        # Load images
        prev_img = cv2.imread(prev_path)
        curr_img = cv2.imread(curr_path)
        
        # Preprocess images
        prev_img = preprocess_image(prev_img, target_size)
        curr_img = preprocess_image(curr_img, target_size)
        
        # Apply data augmentation if requested
        if augment:
            # Apply the same random augmentation to both images
            # Flip horizontally with 50% probability
            if np.random.rand() > 0.5:
                prev_img = np.fliplr(prev_img)
                curr_img = np.fliplr(curr_img)
            
            # Random brightness adjustment (small amount to maintain pair consistency)
            brightness_factor = np.random.uniform(0.9, 1.1)
            prev_img = np.clip(prev_img * brightness_factor, 0, 1)
            curr_img = np.clip(curr_img * brightness_factor, 0, 1)
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(0.9, 1.1)
            mean = np.mean(prev_img, axis=(0, 1), keepdims=True)
            prev_img = np.clip((prev_img - mean) * contrast_factor + mean, 0, 1)
            mean = np.mean(curr_img, axis=(0, 1), keepdims=True)
            curr_img = np.clip((curr_img - mean) * contrast_factor + mean, 0, 1)
        
        # Stack images along channel dimension
        stacked_img = np.concatenate([prev_img, curr_img], axis=-1)
        
        # Get corresponding pose change
        pose = pose_changes[idx]
        
        return stacked_img, pose
    
    # Create dataset from indices
    indices = np.arange(len(image_paths))
    dataset = tf.data.Dataset.from_tensor_slices(indices)
    
    # Map loading function
    dataset = dataset.map(
        lambda idx: tf.py_function(
            load_and_preprocess, 
            [idx], 
            [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # Set output shapes and types
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, (target_size[0], target_size[1], 6)),
            tf.ensure_shape(y, (6,))
        )
    )
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000))
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset, len(image_paths)


def visualize_flow(flow, output_path=None):
    """
    Visualize optical flow using HSV color coding
    
    Args:
        flow: Optical flow field (H, W, 2)
        output_path: Path to save visualization (optional)
        
    Returns:
        Flow visualization as RGB image
    """
    # Convert flow to polar coordinates (magnitude and angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # Set hue according to flow direction
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Set saturation to maximum
    hsv[..., 1] = 255
    
    # Set value according to flow magnitude
    # Normalize magnitude for better visualization
    max_mag = np.max(magnitude)
    if max_mag > 0:
        hsv[..., 2] = np.minimum(magnitude * 255 / max_mag, 255)
    else:
        hsv[..., 2] = 0
    
    # Convert to RGB
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Save if output path is provided
    if output_path is not None:
        cv2.imwrite(output_path, cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))
    
    return flow_rgb


def plot_pose_distribution(csv_path, output_dir=None):
    """
    Plot distribution of pose changes in dataset
    
    Args:
        csv_path: Path to CSV file with pose data
        output_dir: Directory to save plots (optional)
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Extract pose changes
    pose_components = [
        ('delta_x', 'Translation X (m)'),
        ('delta_y', 'Translation Y (m)'),
        ('delta_z', 'Translation Z (m)'),
        ('delta_yaw', 'Rotation Yaw (rad)'),
        ('delta_roll', 'Rotation Roll (rad)'),
        ('delta_pitch', 'Rotation Pitch (rad)')
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (component, label) in enumerate(pose_components):
        data = df[component].values
        
        # Plot histogram
        axes[i].hist(data, bins=50, alpha=0.7)
        axes[i].set_title(f'{label} Distribution')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Frequency')
        
        # Add statistics
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        stats_text = f'Mean: {mean:.4f}\nStd: {std:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
        axes[i].text(0.95, 0.95, stats_text,
                    verticalalignment='top', horizontalalignment='right',
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pose_distribution.png'))
        plt.close()
    else:
        plt.show()


def split_train_test_by_cameras(csv_path, train_cameras, test_cameras, output_dir=None):
    """
    Split dataset into train and test sets based on camera IDs
    
    Args:
        csv_path: Path to CSV file
        train_cameras: List of camera IDs for training
        test_cameras: List of camera IDs for testing
        output_dir: Directory to save split datasets (optional)
        
    Returns:
        train_df, test_df
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Split by camera ID
    train_df = df[df['camera_id'].isin(train_cameras)].copy()
    test_df = df[df['camera_id'].isin(test_cameras)].copy()
    
    print(f"Training set: {len(train_df)} samples from cameras {train_cameras}")
    print(f"Testing set: {len(test_df)} samples from cameras {test_cameras}")
    
    # Save splits if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train_split.csv')
        test_path = os.path.join(output_dir, 'test_split.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Saved train split to {train_path}")
        print(f"Saved test split to {test_path}")
    
    return train_df, test_df


def analyze_dataset(csv_path, output_dir=None):
    """
    Analyze dataset statistics and create summary visualizations
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save analysis results (optional)
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    total_pairs = len(df)
    cameras = df['camera_id'].unique()
    pairs_per_camera = df.groupby('camera_id').size()
    
    print(f"Dataset Summary for {csv_path}:")
    print(f"Total image pairs: {total_pairs}")
    print(f"Cameras: {sorted(cameras)}")
    print("\nPairs per camera:")
    for camera, count in pairs_per_camera.items():
        print(f"  Camera {camera}: {count} pairs")
    
    # Check for missing files
    missing_files = 0
    for i, row in df.iterrows():
        pair_path = row['image_pair_path']
        prev_path = os.path.join(pair_path, 'prev.png')
        curr_path = os.path.join(pair_path, 'curr.png')
        
        if not os.path.exists(prev_path) or not os.path.exists(curr_path):
            missing_files += 1
    
    print(f"\nMissing image files: {missing_files} pairs")
    
    # Time difference statistics
    time_diffs = df['curr_timestamp'] - df['prev_timestamp']
    print(f"\nTime difference between image pairs:")
    print(f"  Mean: {time_diffs.mean():.4f} seconds")
    print(f"  Min: {time_diffs.min():.4f} seconds")
    print(f"  Max: {time_diffs.max():.4f} seconds")
    
    # Pose change statistics
    pose_components = ['delta_x', 'delta_y', 'delta_z', 'delta_yaw', 'delta_roll', 'delta_pitch']
    
    print("\nPose change statistics:")
    for component in pose_components:
        mean = df[component].mean()
        std = df[component].std()
        min_val = df[component].min()
        max_val = df[component].max()
        
        print(f"  {component}:")
        print(f"    Mean: {mean:.6f}")
        print(f"    Std: {std:.6f}")
        print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Create visualizations if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot pose distributions
        plot_pose_distribution(csv_path, output_dir)
        
        # Plot camera distribution
        plt.figure(figsize=(10, 6))
        pairs_per_camera.plot(kind='bar')
        plt.title('Image Pairs per Camera')
        plt.xlabel('Camera ID')
        plt.ylabel('Number of Pairs')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'camera_distribution.png'))
        plt.close()
        
        # Plot time difference distribution
        plt.figure(figsize=(10, 6))
        plt.hist(time_diffs, bins=50)
        plt.title('Time Difference Between Image Pairs')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_diff_distribution.png'))
        plt.close()
        
        print(f"\nSaved analysis visualizations to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data utilities for Operation Barbarosa')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset')
    parser.add_argument('--data', type=str, help='Path to CSV file with pose data')
    parser.add_argument('--output_dir', type=str, help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    if args.analyze:
        if args.data and args.output_dir:
            print(f"Analyzing dataset: {args.data}")
            print(f"Saving results to: {args.output_dir}")
            
            # Create output directory if it doesn't exist
            import os
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Run analysis functions
            analyze_dataset(args.data, args.output_dir)
        else:
            print("Error: --data and --output_dir are required with --analyze")