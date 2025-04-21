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

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU for inference")
else:
    print("No GPU found. Using CPU for inference")

def preprocess_image_pair(prev_img_path, curr_img_path, target_size=(256, 256)):
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
    
    # Normalize pixel values to [-1, 1]
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

def predict_camera_trajectory(model, X_data, y_data, camera_id, output_dir):
    """
    Predict trajectory for a camera and compare with ground truth
    Modified to produce 2D dumbbell-like plots as requested
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output dimension (4DOF or 6DOF)
    output_dim = y_data.shape[1]
    print(f"Output dimension: {output_dim} (4=XYZΨ, 6=XYZΨΘΦ)")
    
    # Initialize trajectories with zero starting position
    start_pos = np.zeros(output_dim)
    true_trajectory = [np.array(start_pos)]
    pred_trajectory = [np.array(start_pos)]
    
    # Process each image pair
    print(f"Predicting trajectory for Camera {camera_id} ({len(X_data)} frames)...")
    for i, (prev_img_path, curr_img_path) in enumerate(tqdm(X_data)):
        try:
            # Preprocess the image pair
            stacked_imgs = preprocess_image_pair(prev_img_path, curr_img_path)
            stacked_imgs_batch = np.expand_dims(stacked_imgs, axis=0)
            
            # Predict the pose change
            predictions = model.predict(stacked_imgs_batch, verbose=0)
            
            # Handle different model outputs
            if isinstance(predictions, list):
                flow_pred, pose_pred = predictions
                delta_pred = pose_pred[0]
            else:
                delta_pred = predictions[0]
            
            # Get the true delta values
            delta_true = y_data[i]
            
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
    
    # 3. Plot training/validation loss if available
    try:
        model_dir = os.path.dirname(os.path.dirname(output_dir))
        history_path = os.path.join(model_dir, 'models_flownet2pose', 'training_history.csv')
        
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            
            plt.figure(figsize=(12, 6))
            for column in history_df.columns:
                if 'loss' in column:
                    plt.plot(history_df[column], label=column)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)
            
            loss_plot_path = os.path.join(output_dir, 'training_losses.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Training loss plots saved to {loss_plot_path}")
    except Exception as e:
        print(f"Could not generate loss plots: {e}")
    
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

def analyze_prediction_errors(trajectory_data, camera_id, output_dir):
    """
    Analyze prediction errors to diagnose issues with the model
    """
    true_trajectory = trajectory_data['true_trajectory']
    pred_trajectory = trajectory_data['pred_trajectory']
    pos_error = trajectory_data['pos_error']
    
    # Calculate cumulative error
    cumulative_error = np.cumsum(pos_error)
    
    # Check for drift (error growing over time)
    error_slope = np.polyfit(np.arange(len(pos_error)), pos_error, 1)[0]
    drift_detected = error_slope > 0.1  # Arbitrary threshold
    
    # Check for sudden jumps in error
    error_diff = np.diff(pos_error)
    jumps = np.where(np.abs(error_diff) > np.std(error_diff) * 3)[0]
    
    # Analyze movement patterns
    x_movement = np.abs(np.diff(true_trajectory[:, 0]))
    y_movement = np.abs(np.diff(true_trajectory[:, 1]))
    z_movement = np.abs(np.diff(true_trajectory[:, 2]))
    
    # Plot error analysis
    plt.figure(figsize=(15, 10))
    
    # Cumulative error
    plt.subplot(2, 2, 1)
    plt.plot(cumulative_error, 'b-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Cumulative Error (m)')
    plt.title('Cumulative Position Error')
    plt.grid(True)
    
    # Error vs. movement magnitude
    plt.subplot(2, 2, 2)
    movement_mag = np.sqrt(x_movement**2 + y_movement**2 + z_movement**2)
    plt.scatter(movement_mag, pos_error[1:], alpha=0.6)
    plt.xlabel('Movement Magnitude (m)')
    plt.ylabel('Position Error (m)')
    plt.title('Error vs. Movement Magnitude')
    plt.grid(True)
    
    # Error by axis
    plt.subplot(2, 2, 3)
    x_error = np.abs(true_trajectory[:, 0] - pred_trajectory[:, 0])
    y_error = np.abs(true_trajectory[:, 1] - pred_trajectory[:, 1])
    z_error = np.abs(true_trajectory[:, 2] - pred_trajectory[:, 2])
    
    plt.plot(x_error, 'r-', linewidth=2, label='X Error')
    plt.plot(y_error, 'g-', linewidth=2, label='Y Error')
    plt.plot(z_error, 'b-', linewidth=2, label='Z Error')
    plt.xlabel('Frame')
    plt.ylabel('Error by Axis (m)')
    plt.title('Error Components by Axis')
    plt.legend()
    plt.grid(True)
    
    # Histogram of errors
    plt.subplot(2, 2, 4)
    plt.hist(pos_error, bins=30, alpha=0.7)
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    analysis_path = os.path.join(output_dir, f'camera{camera_id}_error_analysis.png')
    plt.savefig(analysis_path)
    plt.close()
    
    # Generate a text report
    report = []
    report.append(f"Error Analysis for Camera {camera_id}")
    report.append(f"----------------------------------------")
    report.append(f"Total frames: {len(true_trajectory)}")
    report.append(f"Mean position error: {np.mean(pos_error):.4f} m")
    report.append(f"Max position error: {np.max(pos_error):.4f} m")
    report.append(f"Error standard deviation: {np.std(pos_error):.4f} m")
    report.append(f"Error slope (drift indicator): {error_slope:.6f}")
    
    if drift_detected:
        report.append(f"WARNING: Significant drift detected. Error is growing over time.")
    
    if len(jumps) > 0:
        report.append(f"Found {len(jumps)} sudden jumps in error at frames: {jumps}")
    
    # Component analysis
    report.append(f"\nError by component:")
    report.append(f"Mean X error: {np.mean(x_error):.4f} m")
    report.append(f"Mean Y error: {np.mean(y_error):.4f} m")
    report.append(f"Mean Z error: {np.mean(z_error):.4f} m")
    
    # Save report
    report_path = os.path.join(output_dir, f'camera{camera_id}_error_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Error analysis saved to {analysis_path}")
    print(f"Error report saved to {report_path}")
    
    return {
        'drift_detected': drift_detected,
        'error_slope': error_slope,
        'jumps': jumps,
        'mean_error': np.mean(pos_error)
    }

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
    
    # Create error comparison plot
    plt.figure(figsize=(12, 8))
    
    for i, (camera_id, trajectory_data) in enumerate(trajectories.items()):
        pos_error = trajectory_data['pos_error']
        color = colors[i % len(colors)]
        
        plt.plot(pos_error, color=color, linewidth=2, 
                label=f'Camera {camera_id} (Mean: {np.mean(pos_error):.4f} m)')
    
    plt.xlabel('Frame')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error Comparison Across Cameras')
    plt.legend()
    plt.grid(True)
    
    # Save the error comparison plot
    error_plot_path = os.path.join(output_dir, 'all_cameras_error_comparison.png')
    plt.tight_layout()
    plt.savefig(error_plot_path)
    plt.close()
    print(f"Error comparison plot saved to {error_plot_path}")

def main():
    print("Starting prediction script...")
    print("Arguments:", sys.argv)

    parser = argparse.ArgumentParser(description='Predict trajectories for multiple cameras')
    parser.add_argument('--model_path', required=True, 
                        help='Path to trained model (combined_model_final.h5)')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing the camera CSV files')
    parser.add_argument('--output_dir', default='camera_trajectories',
                        help='Directory to save output visualizations')
    parser.add_argument('--cameras', default='0,1,2,3,4',
                        help='Comma-separated list of camera IDs to process')
    parser.add_argument('--tum_file', 
                        help='Optional path to a TUM format trajectory file to compare with')
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, compile=False)
    print("Model loaded successfully")
    
    # Parse camera IDs
    camera_ids = [int(x) for x in args.cameras.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        
        # Predict trajectory
        trajectory_data = predict_camera_trajectory(
            model, X_data, y_data, camera_id, args.output_dir
        )
        
        # Analyze errors
        error_analysis = analyze_prediction_errors(
            trajectory_data, camera_id, args.output_dir
        )
        
        # Store the trajectory data
        all_trajectories[camera_id] = trajectory_data
    
    # Create combined plot if we have multiple cameras
    if len(all_trajectories) > 1:
        combine_camera_plots(all_trajectories, args.output_dir)
    
    # If a TUM format file is provided, plot it for comparison
    if args.tum_file and os.path.exists(args.tum_file):
        try:
            print(f"Loading TUM format trajectory from {args.tum_file}...")
            # TUM format: timestamp tx ty tz qx qy qz qw
            tum_data = np.loadtxt(args.tum_file)
            
            # Extract position data (x, y, z)
            tum_trajectory = tum_data[:, 1:4]
            
            # Plot TUM trajectory alongside predicted trajectories
            plt.figure(figsize=(12, 10))
            
            # Plot the TUM trajectory first
            plt.plot(tum_trajectory[:, 0], tum_trajectory[:, 1], 'k-', linewidth=3, label='TUM Reference')
            
            # Plot one camera trajectory for comparison
            if all_trajectories:
                camera_id = list(all_trajectories.keys())[0]
                trajectory_data = all_trajectories[camera_id]
                
                true_traj = trajectory_data['true_trajectory']
                pred_traj = trajectory_data['pred_trajectory']
                
                plt.plot(true_traj[:, 0], true_traj[:, 1], 'b-', linewidth=2, label='Ground Truth')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Predicted')
            
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            plt.title('TUM Reference vs. Predicted Trajectory')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            tum_plot_path = os.path.join(args.output_dir, 'tum_trajectory_comparison.png')
            plt.tight_layout()
            plt.savefig(tum_plot_path)
            plt.close()
            print(f"TUM comparison plot saved to {tum_plot_path}")
            
        except Exception as e:
            print(f"Error processing TUM file: {e}")
    
    print("\nAll camera trajectories processed successfully!")