"""
Prediction script for underwater visual odometry model.
Generates trajectory predictions and visualizations.
Part of Operation Barbarosa pipeline.
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pyquaternion import Quaternion

from combined_model import DataNormalizer
from train import load_image_pair

def load_model_and_normalizer(model_path, normalizer_path):
    """
    Load the trained model and data normalizer
    
    Args:
        model_path: Path to the trained model
        normalizer_path: Path to the saved normalizer parameters
        
    Returns:
        model, normalizer
    """
    # Load model
    model = tf.keras.models.load_model(
        model_path, 
        # Custom loss functions will be recompiled during loading
        compile=False
    )
    
    # Compile model with MSE loss for inference
    model.compile(optimizer='adam', loss='mse')
    
    # Load normalizer
    normalizer = DataNormalizer()
    normalizer.load(normalizer_path)
    
    return model, normalizer

def predict_trajectory(model, dataframe, normalizer=None, image_size=(256, 256)):
    """
    Predict trajectory for all image pairs in the dataframe
    
    Args:
        model: Trained model
        dataframe: DataFrame with image pairs
        normalizer: DataNormalizer for denormalizing predictions
        image_size: Input image size
        
    Returns:
        DataFrame with original data and predictions
    """
    # Copy dataframe to avoid modifying the original
    df = dataframe.copy()
    
    # Initialize arrays for predictions
    predictions = np.zeros((len(df), 6))
    
    # Process each image pair
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting trajectories"):
        # Get image paths
        pair_path = row['image_pair_path']
        prev_img_path = os.path.join(pair_path, 'prev.png')
        curr_img_path = os.path.join(pair_path, 'current.png')  # Changed from 'curr.png'
        
        # Load image pair
        try:
            stacked_img = load_image_pair(prev_img_path, curr_img_path, image_size)
            
            # Reshape for model input (add batch dimension)
            model_input = np.expand_dims(stacked_img, axis=0)
            
            # Predict pose changes
            pred = model.predict(model_input, verbose=0)[0]
            
            # Denormalize if necessary
            if normalizer is not None:
                pred = normalizer.denormalize(np.expand_dims(pred, axis=0))[0]
            
            predictions[i] = pred
            
        except Exception as e:
            print(f"Error processing pair {pair_path}: {e}")
            # Use zeros if there's an error
            predictions[i] = np.zeros(6)
    
    # Add predictions to dataframe
    df['pred_delta_x'] = predictions[:, 0]
    df['pred_delta_y'] = predictions[:, 1]
    df['pred_delta_z'] = predictions[:, 2]
    df['pred_delta_yaw'] = predictions[:, 3]
    df['pred_delta_roll'] = predictions[:, 4]
    df['pred_delta_pitch'] = predictions[:, 5]
    
    return df

def calculate_trajectory(dataframe, use_predicted=True, start_pose=None):
    """
    Calculate cumulative trajectory from pose changes
    
    Args:
        dataframe: DataFrame with pose changes
        use_predicted: Whether to use predicted or ground truth changes
        start_pose: Initial pose as [x, y, z, qw, qx, qy, qz]
        
    Returns:
        Array of trajectory poses
    """
    # Sort by pair number to ensure correct order
    df = dataframe.sort_values('pair_num').reset_index(drop=True)
    
    # Initialize trajectory
    n_poses = len(df) + 1  # Include starting pose
    trajectory = np.zeros((n_poses, 7))  # [x, y, z, qw, qx, qy, qz]
    
    # Set initial pose
    if start_pose is not None:
        trajectory[0] = start_pose
    else:
        # Default to identity rotation
        trajectory[0, 3] = 1.0
    
    # Initialize quaternion
    current_quat = Quaternion(trajectory[0, 3:])
    
    # Calculate trajectory
    for i in range(len(df)):
        # Get pose changes (either predicted or ground truth)
        if use_predicted:
            # Use dictionary-style access instead of attribute access
            dx = df.iloc[i]['pred_delta_x']
            dy = df.iloc[i]['pred_delta_y']
            dz = df.iloc[i]['pred_delta_z']
            dyaw = df.iloc[i]['pred_delta_yaw']
            droll = df.iloc[i]['pred_delta_roll']
            dpitch = df.iloc[i]['pred_delta_pitch']
        else:
            dx = df.iloc[i]['delta_x']
            dy = df.iloc[i]['delta_y']
            dz = df.iloc[i]['delta_z']
            dyaw = df.iloc[i]['delta_yaw']
            droll = df.iloc[i]['delta_roll']
            dpitch = df.iloc[i]['delta_pitch']
        
        # Update position (apply rotation to delta position)
        delta_pos = np.array([dx, dy, dz])
        rotated_delta = current_quat.rotate(delta_pos)
        
        trajectory[i+1, 0:3] = trajectory[i, 0:3] + rotated_delta
        
        # Update orientation
        delta_quat = Quaternion(axis=[1, 0, 0], angle=droll) * \
                     Quaternion(axis=[0, 1, 0], angle=dpitch) * \
                     Quaternion(axis=[0, 0, 1], angle=dyaw)
        
        current_quat = current_quat * delta_quat
        trajectory[i+1, 3:] = current_quat.elements
    
    return trajectory

def plot_trajectories(gt_trajectory, pred_trajectory, camera_id, output_path):
    """
    Plot ground truth and predicted trajectories using only 2D plots
    """
    plt.figure(figsize=(15, 10))
    
    # X-Y plot (top-down view)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', label='Ground Truth')
    ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-', label='Predicted')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Camera {camera_id} - X-Y Plane (Top-down View)')
    ax1.grid(True)
    ax1.legend()
    
    # X-Z plot (side view)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', label='Ground Truth')
    ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r-', label='Predicted')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('X-Z Plane (Side View)')
    ax2.grid(True)
    ax2.legend()
    
    # Y-Z plot (front view)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(gt_trajectory[:, 1], gt_trajectory[:, 2], 'b-', label='Ground Truth')
    ax3.plot(pred_trajectory[:, 1], pred_trajectory[:, 2], 'r-', label='Predicted')
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Y-Z Plane (Front View)')
    ax3.grid(True)
    ax3.legend()
    
    # Plot position error over trajectory length
    ax4 = plt.subplot(2, 2, 4)
    position_error = np.linalg.norm(gt_trajectory[:, 0:3] - pred_trajectory[:, 0:3], axis=1)
    ax4.plot(range(len(position_error)), position_error, 'g-')
    ax4.set_xlabel('Trajectory Point')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Position Error')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_trajectory_error(gt_trajectory, pred_trajectory):
    """
    Calculate error metrics between ground truth and predicted trajectories
    
    Args:
        gt_trajectory: Ground truth trajectory
        pred_trajectory: Predicted trajectory
        
    Returns:
        Dictionary of error metrics
    """
    # Ensure both trajectories have the same length
    min_len = min(len(gt_trajectory), len(pred_trajectory))
    gt_trajectory = gt_trajectory[:min_len]
    pred_trajectory = pred_trajectory[:min_len]
    
    # Calculate position error
    position_error = np.linalg.norm(gt_trajectory[:, 0:3] - pred_trajectory[:, 0:3], axis=1)
    
    # Calculate orientation error (quaternion distance)
    orientation_error = []
    for i in range(min_len):
        gt_quat = Quaternion(gt_trajectory[i, 3:])
        pred_quat = Quaternion(pred_trajectory[i, 3:])
        # Ensure quaternions are normalized
        gt_quat = gt_quat.normalised
        pred_quat = pred_quat.normalised
        # Calculate quaternion distance (angle in radians)
        dot_product = np.abs(np.sum(gt_quat.elements * pred_quat.elements))
        dot_product = min(dot_product, 1.0)  # Prevent numerical errors
        angle_error = 2 * np.arccos(dot_product)
        orientation_error.append(angle_error)
    
    orientation_error = np.array(orientation_error)
    
    # Calculate error metrics
    metrics = {
        'mean_position_error': np.mean(position_error),
        'max_position_error': np.max(position_error),
        'mean_orientation_error_deg': np.mean(orientation_error) * (180.0 / np.pi),
        'max_orientation_error_deg': np.max(orientation_error) * (180.0 / np.pi),
        'final_position_error': position_error[-1],
        'final_orientation_error_deg': orientation_error[-1] * (180.0 / np.pi)
    }
    
    return metrics

def process_camera(camera_id, model, normalizer, data_dir, output_dir, image_size=(256, 256)):
    """
    Process a single camera's data
    """
    # Load camera data
    camera_csv = os.path.join(data_dir, f'image_pairs_cam{camera_id}_gt.csv')
    
    if not os.path.exists(camera_csv):
        print(f"Warning: Camera {camera_id} data not found at {camera_csv}")
        return None
    
    print(f"\nProcessing Camera {camera_id}...")
    df = pd.read_csv(camera_csv)
    print(f"Loaded {len(df)} image pairs")
    
    # Create output directory for this camera
    camera_output_dir = os.path.join(output_dir, f'camera{camera_id}')
    os.makedirs(camera_output_dir, exist_ok=True)
    
    # Predict trajectory
    df_with_predictions = predict_trajectory(model, df, normalizer, image_size)
    
    # Save predictions
    predictions_csv = os.path.join(camera_output_dir, 'predictions.csv')
    df_with_predictions.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")
    
    # Calculate trajectories - Use df_with_predictions instead of df
    gt_trajectory = calculate_trajectory(df_with_predictions, use_predicted=False)
    pred_trajectory = calculate_trajectory(df_with_predictions, use_predicted=True)
    
    # Save trajectory data
    np.save(os.path.join(camera_output_dir, 'gt_trajectory.npy'), gt_trajectory)
    np.save(os.path.join(camera_output_dir, 'pred_trajectory.npy'), pred_trajectory)
    
    # Plot trajectories
    plot_path = os.path.join(camera_output_dir, 'trajectory_comparison.png')
    plot_trajectories(gt_trajectory, pred_trajectory, camera_id, plot_path)
    print(f"Saved trajectory plot to {plot_path}")
    
    # Calculate errors
    error_metrics = calculate_trajectory_error(gt_trajectory, pred_trajectory)
    
    # Save metrics
    metrics_path = os.path.join(camera_output_dir, 'error_metrics.npy')
    np.save(metrics_path, error_metrics)
    
    # Print summary
    print("\nError Metrics Summary:")
    for key, value in error_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return error_metrics


def main(args):
    """
    Main function for prediction script
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and normalizer
    normalizer_path = os.path.join(os.path.dirname(args.model_path), 'normalizer.npy')
    model, normalizer = load_model_and_normalizer(args.model_path, normalizer_path)
    
    print(f"Loaded model from {args.model_path}")
    print(f"Loaded normalizer (method: {normalizer.method})")
    
    # Process each camera
    camera_ids = [int(cam) for cam in args.cameras.split(',')]
    print(f"Processing cameras: {camera_ids}")
    
    all_metrics = {}
    
    for camera_id in camera_ids:
        metrics = process_camera(
            camera_id,
            model,
            normalizer,
            args.data_dir,
            args.output_dir,
            image_size=(args.image_height, args.image_width)
        )
        
        if metrics is not None:
            all_metrics[f'camera{camera_id}'] = metrics
    
    # Save combined metrics
    combined_metrics_path = os.path.join(args.output_dir, 'all_camera_metrics.npy')
    np.save(combined_metrics_path, all_metrics)
    
    # Calculate average metrics across all cameras
    if all_metrics:
        mean_position_errors = [m['mean_position_error'] for m in all_metrics.values()]
        mean_orientation_errors = [m['mean_orientation_error_deg'] for m in all_metrics.values()]
        
        avg_metrics = {
            'avg_mean_position_error': np.mean(mean_position_errors),
            'avg_mean_orientation_error_deg': np.mean(mean_orientation_errors)
        }
        
        print("\nAverage Metrics Across All Cameras:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save average metrics
        avg_metrics_path = os.path.join(args.output_dir, 'average_metrics.npy')
        np.save(avg_metrics_path, avg_metrics)
    
    print(f"\nAll prediction results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict trajectories with underwater visual odometry model')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with camera data CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prediction results')
    parser.add_argument('--cameras', type=str, required=True, help='Comma-separated list of camera IDs to process')
    
    # Image parameters
    parser.add_argument('--image_width', type=int, default=256, help='Image width for model input')
    parser.add_argument('--image_height', type=int, default=256, help='Image height for model input')
    
    # Normalization
    parser.add_argument('--normalize_method', type=str, default=None, 
                        help='Override normalizer method (if not using saved normalizer)')
    
    args = parser.parse_args()
    main(args)