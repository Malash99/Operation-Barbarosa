#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import argparse
from tqdm import tqdm

# Print startup message
print("Starting scaled trajectory prediction script...")

def normalize_motion_parameters(data, stats=None, inverse=False):
    """
    Normalize or denormalize motion parameters.
    
    Args:
        data: Numpy array of motion parameters [delta_x, delta_y, delta_z, delta_yaw, delta_pitch, delta_roll]
        stats: Dictionary with 'mean' and 'std' for each parameter (or None to calculate)
        inverse: Boolean, if True performs denormalization instead
        
    Returns:
        Normalized/denormalized data and statistics dictionary
    """
    # Make a copy to avoid modifying the input
    result = data.copy()
    
    # If stats are not provided, calculate them (only needed for normalization)
    if stats is None and not inverse:
        # Calculate stats for each component
        stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0) + 1e-7  # Add small epsilon to avoid division by zero
        }
        
        # Special handling for angular components (periodic values)
        # For 4DOF case - just yaw
        if data.shape[1] >= 4:
            stats['mean'][3] = 0  # Set mean of yaw to 0
            sin_yaw = np.sin(data[:, 3])
            cos_yaw = np.cos(data[:, 3])
            stats['std'][3] = np.sqrt(np.var(sin_yaw) + np.var(cos_yaw))
        
        # For 6DOF case - yaw, pitch, roll
        if data.shape[1] >= 6:
            stats['mean'][4:6] = 0  # Set mean of pitch and roll to 0
            
            sin_pitch = np.sin(data[:, 4])
            cos_pitch = np.cos(data[:, 4])
            stats['std'][4] = np.sqrt(np.var(sin_pitch) + np.var(cos_pitch))
            
            sin_roll = np.sin(data[:, 5])
            cos_roll = np.cos(data[:, 5])
            stats['std'][5] = np.sqrt(np.var(sin_roll) + np.var(cos_roll))
    
    # Perform normalization or denormalization
    if inverse:
        # Denormalize: scaled_value * std + mean
        result = data * stats['std'] + stats['mean']
    else:
        # Normalize: (value - mean) / std
        result = (data - stats['mean']) / stats['std']
    
    return result, stats

def preprocess_image_pair(prev_img_path, curr_img_path, target_size=(256, 256), normalize_method='centered'):
    """
    Load and preprocess an image pair for the model
    """
    # Read images
    prev_img = cv2.imread(prev_img_path)
    curr_img = cv2.imread(curr_img_path)
    
    if prev_img is None or curr_img is None:
        raise ValueError(f"Failed to load images: {prev_img_path} or {curr_img_path}")
    
    # Resize images
    prev_img = cv2.resize(prev_img, target_size)
    curr_img = cv2.resize(curr_img, target_size)
    
    # Convert to RGB if loaded as BGR
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    
    # Apply normalization based on selected method
    if normalize_method == 'meanstd':
        # Normalize using mean and standard deviation (zero mean, unit variance)
        prev_img = prev_img.astype(np.float32)
        curr_img = curr_img.astype(np.float32)
        
        # Calculate mean and std for each image separately
        prev_mean = np.mean(prev_img, axis=(0, 1))
        prev_std = np.std(prev_img, axis=(0, 1)) + 1e-7  # Add small epsilon to avoid division by zero
        
        curr_mean = np.mean(curr_img, axis=(0, 1))
        curr_std = np.std(curr_img, axis=(0, 1)) + 1e-7
        
        # Normalize
        prev_img = (prev_img - prev_mean) / prev_std
        curr_img = (curr_img - curr_mean) / curr_std
        
    elif normalize_method == 'minmax':
        # Normalize to [0, 1] range
        prev_img = prev_img.astype(np.float32) / 255.0
        curr_img = curr_img.astype(np.float32) / 255.0
    else:  # 'centered' (default)
        # Center to [-1, 1] range
        prev_img = prev_img.astype(np.float32) / 127.5 - 1.0
        curr_img = curr_img.astype(np.float32) / 127.5 - 1.0
    
    # Stack the images along the channel dimension
    stacked_imgs = np.concatenate([prev_img, curr_img], axis=-1)
    
    return stacked_imgs

def load_camera_data(csv_path):
    """
    Load camera data from CSV file
    """
    # Read the CSV file
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} image pairs from {csv_path}")
    
    # Extract image paths and ground truth poses
    X_data = []
    y_data = []
    
    # Check if we have delta_pitch and delta_roll columns
    has_pitch_roll = 'delta_pitch' in data.columns and 'delta_roll' in data.columns
    
    for _, row in data.iterrows():
        pair_path = row['image_pair_path']
        prev_path = os.path.join(pair_path, 'prev.png')
        curr_path = os.path.join(pair_path, 'current.png')
        
        if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
            continue
        
        X_data.append((prev_path, curr_path))
        
        # Create target vector based on available data
        if has_pitch_roll:
            y_data.append([
                float(row['delta_x']),
                float(row['delta_y']), 
                float(row['delta_z']),
                float(row['delta_yaw']),
                float(row['delta_pitch']),
                float(row['delta_roll'])
            ])
        else:
            y_data.append([
                float(row['delta_x']),
                float(row['delta_y']), 
                float(row['delta_z']),
                float(row['delta_yaw'])
            ])
    
    return X_data, np.array(y_data)

def load_normalization_stats(model_path):
    """
    Try to load normalization statistics from the model directory
    """
    try:
        model_dir = os.path.dirname(model_path)
        test_data_path = os.path.join(model_dir, 'test_data.npy')
        
        if os.path.exists(test_data_path):
            test_data = np.load(test_data_path, allow_pickle=True).item()
            if 'norm_info' in test_data and 'motion_stats' in test_data['norm_info']:
                norm_info = test_data['norm_info']
                print("Loaded normalization statistics from saved model data")
                return norm_info
    except Exception as e:
        print(f"Error loading normalization stats: {e}")
    
    print("No normalization statistics found, will estimate scaling factors")
    return None

def estimate_scaling_factor(model, X_sample, y_sample, normalize_method='centered'):
    """
    Estimate a scaling factor to correct prediction scale differences
    """
    print("Estimating scaling factor from sample predictions...")
    pred_deltas = []
    
    # Process a few sample image pairs
    for i, (prev_img_path, curr_img_path) in enumerate(X_sample):
        # Preprocess the image pair
        stacked_imgs = preprocess_image_pair(prev_img_path, curr_img_path, normalize_method=normalize_method)
        stacked_imgs_batch = np.expand_dims(stacked_imgs, axis=0)
        
        # Predict the pose change
        predictions = model.predict(stacked_imgs_batch, verbose=0)
        
        # Handle different model outputs
        if isinstance(predictions, list):
            flow_pred, pose_pred = predictions
            delta_pred = pose_pred[0]
        else:
            delta_pred = predictions[0]
        
        pred_deltas.append(delta_pred)
    
    # Convert to numpy array
    pred_deltas = np.array(pred_deltas)
    
    # Calculate average scaling factors per dimension
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    scales = []
    
    for dim in range(y_sample.shape[1]):
        # Get mean absolute values
        true_mean = np.mean(np.abs(y_sample[:, dim]))
        pred_mean = np.mean(np.abs(pred_deltas[:, dim]))
        
        # Calculate scaling factor (true/pred)
        scale = true_mean / (pred_mean + epsilon)
        scales.append(scale)
    
    print(f"Estimated scaling factors: {scales}")
    return scales

def predict_camera_trajectory(model, X_data, y_data, camera_id, output_dir, 
                             scaling_factors=None, norm_info=None, normalize_method='centered'):
    """
    Predict trajectory for a camera and compare with ground truth
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output dimension (4DOF or 6DOF)
    output_dim = y_data.shape[1]
    print(f"Output dimension: {output_dim} (4=XYZΨ, 6=XYZΨΘΦ)")
    
    # Initialize trajectories with zero starting position
    start_pos = np.zeros(output_dim)
    true_trajectory = [np.array(start_pos)]
    pred_trajectory = [np.array(start_pos)]
    
    # Print normalization information if available
    if norm_info is not None and 'motion_stats' in norm_info:
        print(f"Using normalization stats - Mean: {norm_info['motion_stats']['mean']}")
        print(f"Using normalization stats - Std: {norm_info['motion_stats']['std']}")
    elif scaling_factors is not None:
        print(f"Using estimated scaling factors: {scaling_factors}")
    
    # Process each image pair
    print(f"Predicting trajectory for Camera {camera_id} ({len(X_data)} frames)...")
    for i, (prev_img_path, curr_img_path) in enumerate(tqdm(X_data)):
        try:
            # Preprocess the image pair
            stacked_imgs = preprocess_image_pair(
                prev_img_path, curr_img_path, normalize_method=normalize_method)
            stacked_imgs_batch = np.expand_dims(stacked_imgs, axis=0)
            
            # Predict the pose change
            predictions = model.predict(stacked_imgs_batch, verbose=0)
            
            # Handle different model outputs
            if isinstance(predictions, list):
                flow_pred, pose_pred = predictions
                delta_pred = pose_pred[0]
            else:
                delta_pred = predictions[0]
            
            # Store original prediction for debugging
            original_pred = delta_pred.copy()
            
            # Apply denormalization if we have normalization info
            if norm_info is not None and 'motion_stats' in norm_info:
                delta_pred_denorm, _ = normalize_motion_parameters(
                    np.array([delta_pred]), stats=norm_info['motion_stats'], inverse=True)
                delta_pred = delta_pred_denorm[0]
                
                # Debug print for first few predictions
                if i < 3:
                    print(f"Frame {i} - Original prediction: {original_pred}")
                    print(f"Frame {i} - Denormalized prediction: {delta_pred}")
            
            # Apply scaling correction if provided and no normalization info
            elif scaling_factors is not None:
                for j in range(output_dim):
                    delta_pred[j] *= scaling_factors[j]
                
                # Debug print for first few predictions
                if i < 3:
                    print(f"Frame {i} - Original prediction: {original_pred}")
                    print(f"Frame {i} - Scaled prediction: {delta_pred}")
            
            # Get the true delta values
            delta_true = y_data[i]
            
            # Debug print for first few frames
            if i < 3:
                print(f"Frame {i} - Ground truth: {delta_true}")
            
            # Update predicted position
            current_pos = pred_trajectory[-1].copy()
            for j in range(output_dim):
                current_pos[j] += delta_pred[j]
            pred_trajectory.append(current_pos)
            
            # Update ground truth position
            current_pos_true = true_trajectory[-1].copy()
            for j in range(output_dim):
                current_pos_true[j] += delta_true[j]
            true_trajectory.append(current_pos_true)
            
        except Exception as e:
            print(f"Error processing frame {i+1}: {e}")
            continue
    
    # Convert to numpy arrays
    true_trajectory = np.array(true_trajectory)
    pred_trajectory = np.array(pred_trajectory)
    
    # Calculate error metrics
    pos_error = np.sqrt(np.sum((true_trajectory[:, :3] - pred_trajectory[:, :3])**2, axis=1))
    mean_pos_error = np.mean(pos_error)
    final_pos_error = pos_error[-1] if len(pos_error) > 0 else 0
    
    # Handle angular errors based on dimensions
    if output_dim > 4:  # 6DOF case
        yaw_error = np.abs(np.mod(true_trajectory[:, 3] - pred_trajectory[:, 3] + np.pi, 2*np.pi) - np.pi)
        mean_yaw_error = np.mean(yaw_error)
    else:  # 4DOF case
        yaw_error = np.abs(np.mod(true_trajectory[:, 3] - pred_trajectory[:, 3] + np.pi, 2*np.pi) - np.pi)
        mean_yaw_error = np.mean(yaw_error)
    
    # 1. 2D Trajectory Plot (dumbbell-like view as requested)
    plt.figure(figsize=(12, 10))
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', linewidth=2, label='Ground Truth')
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, label='Predicted')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title(f'Camera {camera_id} Trajectory Path')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'camera{camera_id}_trajectory_2d.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"2D trajectory plot saved to {plot_path}")
    
    # 2. Plot errors over time
    plt.figure(figsize=(15, 6))
    
    # Position error
    plt.subplot(1, 2, 1)
    plt.plot(pos_error, 'g-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Position Error (m)')
    plt.title(f'Position Error Over Time (Mean: {mean_pos_error:.4f} m)')
    plt.grid(True)
    
    # Yaw error
    plt.subplot(1, 2, 2)
    plt.plot(yaw_error, 'm-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Yaw Error (rad)')
    plt.title(f'Yaw Error Over Time (Mean: {mean_yaw_error:.4f} rad)')
    plt.grid(True)
    
    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, f'camera{camera_id}_errors.png')
    plt.savefig(error_plot_path)
    plt.close()
    print(f"Error plots saved to {error_plot_path}")
    
    # Save trajectory data
    trajectory_data = {
        'true_trajectory': true_trajectory,
        'pred_trajectory': pred_trajectory,
        'pos_error': pos_error,
        'yaw_error': yaw_error
    }
    
    # Save as CSV for easier analysis
    df = pd.DataFrame({
        'frame': range(len(true_trajectory)),
        'true_x': true_trajectory[:, 0],
        'true_y': true_trajectory[:, 1],
        'true_z': true_trajectory[:, 2],
        'true_yaw': true_trajectory[:, 3],
        'pred_x': pred_trajectory[:, 0],
        'pred_y': pred_trajectory[:, 1],
        'pred_z': pred_trajectory[:, 2],
        'pred_yaw': pred_trajectory[:, 3],
        'pos_error': pos_error,
        'yaw_error': yaw_error
    })
    
    if output_dim > 4:
        df['true_pitch'] = true_trajectory[:, 4]
        df['true_roll'] = true_trajectory[:, 5]
        df['pred_pitch'] = pred_trajectory[:, 4]
        df['pred_roll'] = pred_trajectory[:, 5]
    
    csv_path = os.path.join(output_dir, f'camera{camera_id}_trajectory.csv')
    df.to_csv(csv_path, index=False)
    print(f"Trajectory data saved to {csv_path}")
    
    # Print summary statistics
    print(f"\nCamera {camera_id} Trajectory Prediction Results:")
    print(f"Total trajectory points: {len(true_trajectory)}")
    print(f"Mean position error: {mean_pos_error:.4f} m")
    print(f"Mean yaw error: {mean_yaw_error:.4f} rad")
    print(f"Final position error: {final_pos_error:.4f} m")
    
    return trajectory_data

def combine_camera_plots(trajectories, output_dir):
    """
    Create a single 2D plot combining all camera trajectories
    """
    # 2D plot (XY plane - top view)
    plt.figure(figsize=(12, 10))
    
    colors = ['b', 'g', 'r', 'm', 'c']
    line_styles = ['-', '--']
    
    for i, (camera_id, trajectory_data) in enumerate(trajectories.items()):
        true_traj = trajectory_data['true_trajectory']
        pred_traj = trajectory_data['pred_trajectory']
        
        color = colors[i % len(colors)]
        
        # Ground truth with solid line
        plt.plot(true_traj[:, 0], true_traj[:, 1], 
                f'{color}{line_styles[0]}', linewidth=2, 
                label=f'Camera {camera_id} (Ground Truth)')
        
        # Prediction with dashed line
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 
                f'{color}{line_styles[1]}', linewidth=2, 
                label=f'Camera {camera_id} (Predicted)')
    
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('All Camera Trajectories - 2D View')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    combined_plot_path = os.path.join(output_dir, 'all_cameras_trajectories_2d.png')
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"Combined 2D plot saved to {combined_plot_path}")

def main():
    print("\nStarting trajectory prediction with normalization handling...")

    parser = argparse.ArgumentParser(description='Predict trajectories with normalization handling')
    parser.add_argument('--model_path', required=True, 
                        help='Path to trained model (combined_model_final.h5)')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing the camera CSV files')
    parser.add_argument('--output_dir', default='camera_test_results',
                        help='Directory to save output visualizations')
    parser.add_argument('--cameras', default='1,2,3,4',
                        help='Comma-separated list of camera IDs to process')
    parser.add_argument('--scaling_sample_size', type=int, default=20,
                        help='Number of samples to use for scale estimation')
    parser.add_argument('--normalize_method', choices=['minmax', 'meanstd', 'centered'], 
                      default='centered', help='Image normalization method')
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, compile=False)
    print("Model loaded successfully")
    
    # Try to load normalization statistics from model directory
    norm_info = load_normalization_stats(args.model_path)
    
    # Parse camera IDs
    camera_ids = [int(x) for x in args.cameras.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Estimate scaling factors if normalization info is not available
    scaling_factors = None
    if norm_info is None:
        # Process first camera to estimate scaling factors
        first_camera_id = camera_ids[0]
        print(f"\nUsing camera {first_camera_id} to estimate scaling factors...")
        
        csv_path = os.path.join(args.data_dir, f'image_pairs_cam{first_camera_id}_gt.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file for camera {first_camera_id} not found at {csv_path}")
            return
        
        # Load camera data
        X_data, y_data = load_camera_data(csv_path)
        
        if len(X_data) == 0:
            print(f"Warning: No valid image pairs found for camera {first_camera_id}")
            return
        
        # Use a sample to estimate scaling factors
        sample_size = min(args.scaling_sample_size, len(X_data))
        X_sample = X_data[:sample_size]
        y_sample = y_data[:sample_size]
        
        # Estimate scaling factors
        scaling_factors = estimate_scaling_factor(model, X_sample, y_sample, 
                                                 normalize_method=args.normalize_method)
    
    # Process each camera
    all_trajectories = {}
    
    for camera_id in camera_ids:
        csv_path = os.path.join(args.data_dir, f'image_pairs_cam{camera_id}_gt.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file for camera {camera_id} not found at {csv_path}")
            continue
        
        # Load camera data
        X_data, y_data = load_camera_data(csv_path)
        
        if len(X_data) == 0:
            print(f"Warning: No valid image pairs found for camera {camera_id}")
            continue
        
        # Predict trajectory with proper normalization handling
        trajectory_data = predict_camera_trajectory(
            model, X_data, y_data, camera_id, args.output_dir,
            scaling_factors=scaling_factors,
            norm_info=norm_info,
            normalize_method=args.normalize_method
        )
        
        # Store the trajectory data
        all_trajectories[camera_id] = trajectory_data
    
    # Create combined plot if we have multiple cameras
    if len(all_trajectories) > 1:
        combine_camera_plots(all_trajectories, args.output_dir)
    
    print("\nAll camera trajectories processed successfully!")

if __name__ == "__main__":
    main()