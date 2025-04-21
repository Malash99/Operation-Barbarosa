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

print("Starting prediction debugging script...")

# Set up paths
model_path = '/app/output/models_flownet2pose/combined_model_final.h5'
data_dir = '/app/output'
output_dir = '/app/output/camera_test_results'
camera_id = 1

# Create output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

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
missing_count = 0

# Check if we have delta_pitch and delta_roll columns
has_pitch_roll = 'delta_pitch' in data.columns and 'delta_roll' in data.columns
print(f"CSV has pitch/roll data: {has_pitch_roll}")

print("Processing image paths...")
for _, row in data.iterrows():
    pair_path = row['image_pair_path']
    prev_path = os.path.join(pair_path, 'prev.png')
    curr_path = os.path.join(pair_path, 'current.png')
    
    if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
        missing_count += 1
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

print(f"Processed {len(X_data)} valid image pairs, {missing_count} missing pairs")

if len(X_data) == 0:
    print("ERROR: No valid image pairs found! Exiting.")
    sys.exit(1)

# Check the first few image paths to verify they exist
print("\nChecking a few image paths:")
for i in range(min(3, len(X_data))):
    prev_path, curr_path = X_data[i]
    print(f"  Pair {i+1}: {prev_path} (exists: {os.path.exists(prev_path)}), {curr_path} (exists: {os.path.exists(curr_path)})")

# Convert y_data to numpy array
y_data = np.array(y_data)
print(f"\nShape of y_data: {y_data.shape}")

# Create a simple trajectory plot with the first few points of ground truth
print("\nCreating test trajectory plot...")
plt.figure(figsize=(12, 10))
plt.plot(y_data[:20, 0].cumsum(), y_data[:20, 1].cumsum(), 'b-', linewidth=2, label='Ground Truth (first 20 points)')
plt.xlabel('X position (cumulative delta_x)')
plt.ylabel('Y position (cumulative delta_y)')
plt.title(f'Test Camera {camera_id} Trajectory (First 20 Points)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f'test_trajectory_cam{camera_id}.png'))
plt.close()
print(f"Saved test trajectory plot")

print("\nDebug prediction script completed successfully!")
