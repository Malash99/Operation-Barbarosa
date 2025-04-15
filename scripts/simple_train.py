#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
import cv2

# Configuration
DATA_DIR = "/app/output/image_pairs_cam0"
MODEL_DIR = "/app/output/simple_model"
BATCH_SIZE = 4
EPOCHS = 5
IMG_SIZE = (256, 256)

# 1. Data Preparation
def load_sample(pair_dir):
    """Load a single image pair"""
    prev = cv2.resize(cv2.imread(f"{pair_dir}/prev.png"), IMG_SIZE)
    curr = cv2.resize(cv2.imread(f"{pair_dir}/current.png"), IMG_SIZE)
    return np.concatenate([prev, curr], axis=-1)  # Stack along channels

# Get all pairs
pair_dirs = [f"{DATA_DIR}/{d}" for d in os.listdir(DATA_DIR) if d.startswith("pair_")]
print(f"Found {len(pair_dirs)} image pairs")

# 2. Simple Model
inputs = Input(shape=(*IMG_SIZE, 6))  # 6 channels (2 RGB images)
x = Conv2D(8, 3, activation='relu')(inputs)
x = Flatten()(x)
outputs = Dense(4)(x)  # Predict [delta_x, delta_y, delta_z, delta_yaw]
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

# 3. Training
X = np.array([load_sample(p) for p in pair_dirs[:50]])  # Use first 50 samples
y = np.random.rand(50, 4)  # Dummy labels for testing

model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save(f"{MODEL_DIR}/simple_model.h5")
print("Training complete!")