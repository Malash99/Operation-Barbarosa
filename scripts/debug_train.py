import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Starting debug script...")

# Configure TensorFlow to use less memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

# Define a simpler model
def build_simple_model(input_shape=(256, 256, 6)):
    print("Building simplified model...")
    inputs = Input(shape=input_shape)
    
    # Simple encoder
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Simple regression head
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(4, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print("Model compiled successfully")
    
    return model

# Image preprocessing function
def preprocess_image_pair(prev_img_path, curr_img_path, target_size=(256, 256)):
    try:
        # Read images
        prev_img = cv2.imread(prev_img_path)
        curr_img = cv2.imread(curr_img_path)
        
        if prev_img is None or curr_img is None:
            return None
        
        # Resize images
        prev_img = cv2.resize(prev_img, target_size)
        curr_img = cv2.resize(curr_img, target_size)
        
        # Convert to RGB
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        prev_img = prev_img.astype(np.float32) / 127.5 - 1.0
        curr_img = curr_img.astype(np.float32) / 127.5 - 1.0
        
        # Stack images
        stacked_imgs = np.concatenate([prev_img, curr_img], axis=-1)
        
        return stacked_imgs
    except Exception as e:
        print(f'Error preprocessing images: {e}')
        return None

# Load a small subset of data for testing
print("Loading data...")
data = pd.read_csv('/app/output/all_cameras_with_gt.csv')
print(f"Loaded {len(data)} rows")

# Filter by camera ID
train_cameras = [0, 1, 2, 3]
test_cameras = [4]
train_data = data[data['camera_id'].isin(train_cameras)].sample(50)  # Just use 50 samples
test_data = data[data['camera_id'].isin(test_cameras)].sample(10)    # Just use 10 samples

print(f"Using {len(train_data)} training samples and {len(test_data)} test samples")

# Prepare data
X_train = []
y_train = []

print("Preprocessing training images...")
for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
    pair_path = row['image_pair_path']
    prev_path = os.path.join(pair_path, 'prev.png')
    curr_path = os.path.join(pair_path, 'current.png')
    
    stacked_imgs = preprocess_image_pair(prev_path, curr_path)
    if stacked_imgs is not None:
        X_train.append(stacked_imgs)
        y_train.append([
            row['delta_x'], 
            row['delta_y'], 
            row['delta_z'], 
            row['delta_yaw']
        ])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Prepared {len(X_train)} training samples")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Build and train model
model = build_simple_model()

print("Starting training...")
try:
    model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=4,
        verbose=1
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")

print("Script finished!")