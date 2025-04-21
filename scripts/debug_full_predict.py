#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm

def preprocess_image_pair(prev_img_path, curr_img_path, target_size=(256, 256)):
    """Load and preprocess an image pair for the model"""
    print(f"  Preprocessing images: {prev_img_path}, {curr_img_path}")
    
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

print("Starting full prediction debugging script...")

# Set up paths
model_path = '/app/output/models_flownet2pose/combined_model_final.h5'
data_dir = '/app/output'
output_dir = '/app/output/camera_test_results'
camera_id = 1

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load the model
print(f"Loading model from {model_path}...")
model = load_model(model_path, compile=False)
print("Model loaded successfully!")

# Load camera data
csv_path = os.path.join(data_dir, f'image_pairs_cam{camera_id}_gt.csv')
print(f"Loading CSV from {csv_path}...")
data = pd.read_csv(csv_path)
print(f"CSV loaded successfully with {len(data)} rows")

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

print(f"Processed {len(X_data)} valid image pairs")

# Convert to numpy array
y_data = np.array(y_data)
output_dim = y_data.shape[1]
print(f"Output dimension: {output_dim}")

# Try to process the first 5 pairs only for testing
num_pairs_to_process = 5
X_data = X_data[:num_pairs_to_process]
y_data = y_data[:num_pairs_to_process]

# Initialize trajectories with zero starting position
start_pos = np.zeros(output_dim)
true_trajectory = [np.array(start_pos)]
pred_trajectory = [np.array(start_pos)]

# Process each image pair
print(f"\nPredicting trajectory for {len(X_data)} image pairs...")
for i, (prev_img_path, curr_img_path) in enumerate(X_data):
    try:
        print(f"\nProcessing pair {i+1}/{len(X_data)}:")
        # Preprocess the image pair
        stacked_imgs = preprocess_image_pair(prev_img_path, curr_img_path)
        stacked_imgs_batch = np.expand_dims(stacked_imgs, axis=0)
        
        # Predict the pose change
        print("  Running model prediction...")
        predictions = model.predict(stacked_imgs_batch, verbose=0)
        
        # Handle different model outputs
        if isinstance(predictions, list):
            print("  Model returned multiple outputs")
            flow_pred, pose_pred = predictions
            delta_pred = pose_pred[0]
        else:
            print("  Model returned single output")
            delta_pred = predictions[0]
        
        print(f"  Predicted delta: {delta_pred}")
        
        # Get the true delta values
        delta_true = y_data[i]
        print(f"  True delta: {delta_true}")
        
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
        
        print(f"  Updated trajectory, now have {len(pred_trajectory)} points")
        
    except Exception as e:
        print(f"Error processing frame {i+1}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Convert to numpy arrays
true_trajectory = np.array(true_trajectory)
pred_trajectory = np.array(pred_trajectory)

print(f"\nTrue trajectory shape: {true_trajectory.shape}")
print(f"Prediction trajectory shape: {pred_trajectory.shape}")

# Create 2D trajectory plot
print("Creating trajectory plot...")
plt.figure(figsize=(12, 10))
plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', linewidth=2, label='Ground Truth')
plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, label='Predicted')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title(f'Camera {camera_id} Trajectory Path (First {num_pairs_to_process} Frames)')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(output_dir, f'camera{camera_id}_debug_trajectory.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved trajectory plot to {plot_path}")

print("\nFull prediction debug script completed successfully!")
