#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Print debugging information
print("Starting debug script...")

# Load the model
model_path = '/app/output/models_flownet2pose/combined_model_final.h5'
print(f"Attempting to load model from {model_path}...")
model = load_model(model_path, compile=False)
print("Model loaded successfully!")

# Load one camera's data
csv_path = '/app/output/image_pairs_cam1_gt.csv'
print(f"Loading CSV from {csv_path}...")
data = pd.read_csv(csv_path)
print(f"CSV loaded successfully with {len(data)} rows")

# Create output directory
output_dir = '/app/output/camera_test_results'
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Just create a simple plot to verify matplotlib is working
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title('Test Plot')
plt.savefig(os.path.join(output_dir, 'test_plot.png'))
plt.close()
print(f"Created test plot in {output_dir}")

print("Debug script completed successfully!")
