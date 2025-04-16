#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU for training")
else:
    print("No GPU found. Using CPU for training")

def build_flownet_model(input_shape=(256, 256, 6)):
    """
    Build a FlowNet model that predicts optical flow from image pairs
    """
    inputs = Input(shape=input_shape)
    
    # Contracting path (encoder)
    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
    
    conv2 = Conv2D(128, (5, 5), padding='same', activation='relu', name='conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
    
    conv3 = Conv2D(256, (5, 5), padding='same', activation='relu', name='conv3')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
    
    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)
    
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5')(pool4)
    
    # Expanding path (decoder) for flow prediction
    up6 = UpSampling2D(size=(2, 2), name='up6')(conv5)
    concat6 = Concatenate(name='concat6')([up6, conv4])
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6')(concat6)
    
    up7 = UpSampling2D(size=(2, 2), name='up7')(conv6)
    concat7 = Concatenate(name='concat7')([up7, conv3])
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7')(concat7)
    
    up8 = UpSampling2D(size=(2, 2), name='up8')(conv7)
    concat8 = Concatenate(name='concat8')([up8, conv2])
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(concat8)
    
    # Final flow prediction (2 channels for x and y flow)
    flow = Conv2D(2, (3, 3), padding='same', activation='tanh', name='flow_output')(conv8)
    
    # Create the model
    model = Model(inputs=inputs, outputs=flow, name='flownet')
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    
    return model

def build_pose_regression_model(input_shape=(64, 64, 2), output_dim=6):
    """
    Build a pose regression model that takes optical flow as input
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional feature extraction
    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (5, 5), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Flatten and dense layers
    flat = Flatten()(pool3)
    fc1 = Dense(512, activation='relu')(flat)
    bn1 = BatchNormalization()(fc1)
    drop1 = Dropout(0.3)(bn1)
    
    fc2 = Dense(128, activation='relu')(drop1)
    bn2 = BatchNormalization()(fc2)
    drop2 = Dropout(0.3)(bn2)
    
    # Output layer
    outputs = Dense(output_dim, activation='linear', name='pose_output')(drop2)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name='pose_regression')
    
    # Define custom loss function for angular components
    def angular_loss(y_true, y_pred):
        # Position components (first 3)
        pos_true, pos_pred = y_true[:, :3], y_pred[:, :3]
        pos_loss = tf.reduce_mean(tf.square(pos_true - pos_pred))
        
        # Angular components (last 3 if available)
        if output_dim > 3:
            ang_true, ang_pred = y_true[:, 3:], y_pred[:, 3:]
            # Handle circular nature of angles
            ang_diff = tf.math.floormod(ang_true - ang_pred + np.pi, 2 * np.pi) - np.pi
            ang_loss = tf.reduce_mean(tf.square(ang_diff))
            # Weight angular components more heavily
            return pos_loss + 2.0 * ang_loss
        else:
            return pos_loss
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=angular_loss)
    
    return model

def build_combined_model(flownet_model, pose_model, input_shape=(256, 256, 6)):
    """
    Build a combined model that connects FlowNet to pose regression
    """
    # Input for the combined model
    inputs = Input(shape=input_shape)
    
    # Get flow from FlowNet model
    flow = flownet_model(inputs)
    
    # Feed flow to pose regression model
    pose = pose_model(flow)
    
    # Create the combined model
    combined_model = Model(inputs=inputs, outputs=[flow, pose])
    
    # Define custom loss function for angular components
    def angular_loss(y_true, y_pred):
        # Position components (first 3)
        pos_true, pos_pred = y_true[:, :3], y_pred[:, :3]
        pos_loss = tf.reduce_mean(tf.square(pos_true - pos_pred))
        
        # Angular components (last 3 if available)
        if y_true.shape[1] > 3:
            ang_true, ang_pred = y_true[:, 3:], y_pred[:, 3:]
            # Handle circular nature of angles
            ang_diff = tf.math.floormod(ang_true - ang_pred + np.pi, 2 * np.pi) - np.pi
            ang_loss = tf.reduce_mean(tf.square(ang_diff))
            # Weight angular components more heavily
            return pos_loss + 2.0 * ang_loss
        else:
            return pos_loss
    
    # Compile the combined model
    combined_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'flownet': 'mse',
            'pose_regression': angular_loss
        },
        loss_weights={
            'flownet': 0.5,  # Weight for flow prediction
            'pose_regression': 1.0  # Weight for pose prediction
        }
    )
    
    return combined_model

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

def load_dataset_by_camera(dataset_csv, train_cameras=[0, 1, 2, 3], test_cameras=[4], root_dir=None):
    """
    Load the dataset from the CSV file and split by camera ID
    """
    # Read the CSV file ensuring proper types
    data = pd.read_csv(dataset_csv, dtype={'camera_id': int})
    print(f"Loaded {len(data)} image pairs from {dataset_csv}")
    
    # Check if we have delta_pitch and delta_roll columns
    has_pitch_roll = 'delta_pitch' in data.columns and 'delta_roll' in data.columns
    if has_pitch_roll:
        print("Delta pitch and roll data available - using all 6DOF motion parameters")
    else:
        print("Warning: delta_pitch and delta_roll not found in dataset - using only 4DOF motion parameters")
    
    # Debug: Show camera distribution
    print("\nCamera ID distribution in CSV:")
    print(data['camera_id'].value_counts().sort_index())
    
    # Convert camera IDs to integers if they're strings
    train_cameras = [int(cam) for cam in train_cameras]
    test_cameras = [int(cam) for cam in test_cameras]
    
    # Filter data
    train_data = data[data['camera_id'].isin(train_cameras)]
    test_data = data[data['camera_id'].isin(test_cameras)]
    
    print(f"\nTraining data: {len(train_data)} samples from cameras {train_cameras}")
    print(f"Testing data: {len(test_data)} samples from cameras {test_cameras}")
    
    # Process training data
    X_train = []
    y_train = []
    missing_count = 0
    
    for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Loading training data"):
        pair_path = row['image_pair_path']
        prev_path = os.path.join(pair_path, 'prev.png')
        curr_path = os.path.join(pair_path, 'current.png')
        
        if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
            missing_count += 1
            continue
        
        X_train.append((prev_path, curr_path))
        
        # Create target vector based on available data
        if has_pitch_roll:
            y_train.append([
                float(row['delta_x']),
                float(row['delta_y']), 
                float(row['delta_z']),
                float(row['delta_yaw']),
                float(row['delta_pitch']),
                float(row['delta_roll'])
            ])
        else:
            y_train.append([
                float(row['delta_x']),
                float(row['delta_y']), 
                float(row['delta_z']),
                float(row['delta_yaw'])
            ])
    
    if missing_count > 0:
        print(f"Warning: Missing {missing_count} training image pairs")
    
    # Process test data
    X_test = []
    y_test = []
    missing_count = 0
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Loading test data"):
        pair_path = row['image_pair_path']
        prev_path = os.path.join(pair_path, 'prev.png')
        curr_path = os.path.join(pair_path, 'current.png')
        
        if not (os.path.exists(prev_path) and os.path.exists(curr_path)):
            missing_count += 1
            continue
        
        X_test.append((prev_path, curr_path))
        
        # Create target vector based on available data
        if has_pitch_roll:
            y_test.append([
                float(row['delta_x']),
                float(row['delta_y']),
                float(row['delta_z']),
                float(row['delta_yaw']),
                float(row['delta_pitch']),
                float(row['delta_roll'])
            ])
        else:
            y_test.append([
                float(row['delta_x']),
                float(row['delta_y']),
                float(row['delta_z']),
                float(row['delta_yaw'])
            ])
    
    if missing_count > 0:
        print(f"Warning: Missing {missing_count} test image pairs")
    
    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, np.array(y_train),
        test_size=0.2,
        random_state=42
    )
    
    print("\nFinal dataset sizes:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Determine output dimension for the model
    output_dim = 6 if has_pitch_roll else 4
    print(f"Motion parameters dimension: {output_dim}")
    
    return X_train, X_val, X_test, np.array(y_train), np.array(y_val), np.array(y_test), output_dim

def compute_flow_from_model(flownet_model, image_pair):
    """
    Compute optical flow from an image pair using the FlowNet model
    """
    # Predict flow using FlowNet
    flow = flownet_model.predict(np.expand_dims(image_pair, axis=0))[0]
    return flow

def data_generator(image_pairs, targets, flownet_model=None, batch_size=16, 
                   target_size=(256, 256), flow_size=(64, 64), shuffle=True):
    """
    Generator function to yield batches of data
    If flownet_model is provided, it will generate flow for pose_regression_model
    Otherwise, it will generate image pairs for the combined_model
    """
    num_samples = len(image_pairs)
    indices = np.arange(num_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_inputs = []
            batch_targets = []
            
            for idx in batch_indices:
                prev_img_path, curr_img_path = image_pairs[idx]
                
                try:
                    # Preprocess the image pair
                    stacked_imgs = preprocess_image_pair(
                        prev_img_path, curr_img_path, target_size
                    )
                    
                    batch_inputs.append(stacked_imgs)
                    batch_targets.append(targets[idx])
                except Exception as e:
                    print(f"Error processing {prev_img_path}, {curr_img_path}: {e}")
                    continue
            
            if not batch_inputs:
                continue
                
            X_batch = np.array(batch_inputs)
            y_batch = np.array(batch_targets)
            
            if flownet_model is None:
                # For the combined model, need placeholder flow targets
                flow_shape = (X_batch.shape[0], target_size[0]//4, target_size[1]//4, 2)
                dummy_flow = np.zeros(flow_shape)
                yield X_batch, [dummy_flow, y_batch]
            else:
                # For the pose regression model, compute flow
                flows = []
                for img_pair in X_batch:
                    flow = compute_flow_from_model(flownet_model, img_pair)
                    # Resize flow to expected input size for pose model
                    if flow.shape[:2] != flow_size:
                        flow = cv2.resize(flow, flow_size)
                    flows.append(flow)
                
                yield np.array(flows), y_batch

def train_combined_model(model, X_train, X_val, y_train, y_val, batch_size=16, epochs=50, 
                        model_save_path='models'):
    """
    Train the combined FlowNet + Pose Regression model
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create data generators
    train_gen = data_generator(X_train, y_train, batch_size=batch_size)
    val_gen = data_generator(X_val, y_val, batch_size=batch_size)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_save_path, 'combined_model.h5'),
        monitor='val_pose_regression_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_pose_regression_loss',
        patience=10,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_pose_regression_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=val_gen,
        validation_steps=len(X_val) // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_save_path, 'combined_model_final.h5'))
    
    # Save the training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_save_path, 'training_history.csv'), index=False)
    
    return model, history

def train_two_stage(flownet_model, pose_model, X_train, X_val, y_train, y_val, 
                    batch_size=16, epochs=50, model_save_path='models'):
    """
    Train the two-stage model: first FlowNet, then Pose Regression
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    # Step 1: Train FlowNet model (with dummy targets)
    print("Training FlowNet model...")
    flow_train_gen = data_generator(X_train, y_train, batch_size=batch_size)
    flow_val_gen = data_generator(X_val, y_val, batch_size=batch_size)
    
    # Set up callbacks for FlowNet
    flow_checkpoint = ModelCheckpoint(
        os.path.join(model_save_path, 'flownet_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    flow_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )
    
    flow_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train FlowNet
    flow_history = flownet_model.fit(
        flow_train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=flow_val_gen,
        validation_steps=len(X_val) // batch_size,
        epochs=epochs // 2,  # Use fewer epochs for flow
        callbacks=[flow_checkpoint, flow_early_stopping, flow_reduce_lr],
        verbose=1
    )
    
    # Save FlowNet model
    flownet_model.save(os.path.join(model_save_path, 'flownet_model_final.h5'))
    
    # Step 2: Train Pose Regression model using FlowNet output
    print("\nTraining Pose Regression model...")
    pose_train_gen = data_generator(X_train, y_train, flownet_model=flownet_model, 
                                   batch_size=batch_size)
    pose_val_gen = data_generator(X_val, y_val, flownet_model=flownet_model, 
                                 batch_size=batch_size)
    
    # Set up callbacks for Pose Regression
    pose_checkpoint = ModelCheckpoint(
        os.path.join(model_save_path, 'pose_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    pose_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )
    
    pose_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train Pose Regression
    pose_history = pose_model.fit(
        pose_train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=pose_val_gen,
        validation_steps=len(X_val) // batch_size,
        epochs=epochs,
        callbacks=[pose_checkpoint, pose_early_stopping, pose_reduce_lr],
        verbose=1
    )
    
    # Save Pose Regression model
    pose_model.save(os.path.join(model_save_path, 'pose_model_final.h5'))
    
    # Create and save the combined model
    combined_model = build_combined_model(flownet_model, pose_model)
    combined_model.save(os.path.join(model_save_path, 'combined_model_final.h5'))
    
    # Save the training histories
    flow_history_df = pd.DataFrame(flow_history.history)
    flow_history_df.to_csv(os.path.join(model_save_path, 'flow_training_history.csv'), index=False)
    
    pose_history_df = pd.DataFrame(pose_history.history)
    pose_history_df.to_csv(os.path.join(model_save_path, 'pose_training_history.csv'), index=False)
    
    return flownet_model, pose_model, combined_model

def evaluate_model(combined_model, X_test, y_test, batch_size=16):
    """
    Evaluate the model on test data
    """
    # Create a generator for test data
    test_gen = data_generator(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    # Predict on test data
    y_pred = []
    
    for i in range(0, len(X_test), batch_size):
        batch_size_actual = min(batch_size, len(X_test) - i)
        
        # Get a batch of preprocessed images
        batch_inputs = []
        for j in range(batch_size_actual):
            idx = i + j
            if idx >= len(X_test):
                break
                
            prev_img_path, curr_img_path = X_test[idx]
            try:
                stacked_imgs = preprocess_image_pair(prev_img_path, curr_img_path)
                batch_inputs.append(stacked_imgs)
            except Exception as e:
                print(f"Error processing {prev_img_path}, {curr_img_path}: {e}")
                batch_inputs.append(np.zeros((256, 256, 6)))
        
        X_batch = np.array(batch_inputs)
        
        # Predict on the batch
        _, pose_pred = combined_model.predict(X_batch)
        y_pred.extend(pose_pred)
    
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    
    # Calculate metrics
    mse = np.mean(np.square(y_test - y_pred), axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred), axis=0)
    
    print("\nEvaluation metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    
    return y_pred, mse, rmse, mae

def predict_camera_trajectory(combined_model, X_test, y_test, start_pos=None, plot_save_path=None):
    """
    Predict trajectory for a specific camera and compare with ground truth
    """
    # Determine if we're doing 4DOF or 6DOF based on the data
    output_dim = y_test.shape[1]
    
    # Set default start position based on dimensions
    if start_pos is None:
        start_pos = np.zeros(output_dim)
    
    # Initialize trajectories
    true_trajectory = [np.array(start_pos)]
    pred_trajectory = [np.array(start_pos)]
    
    # Process each image pair
    for i, (prev_img_path, curr_img_path) in enumerate(tqdm(X_test, desc="Predicting camera trajectory")):
        try:
            # Preprocess the image pair
            stacked_imgs = preprocess_image_pair(prev_img_path, curr_img_path)
            stacked_imgs = np.expand_dims(stacked_imgs, axis=0)
            
            # Predict the pose change
            _, pose_pred = combined_model.predict(stacked_imgs)
            delta_pred = pose_pred[0]
            
            # Get the true delta values
            delta_true = y_test[i]
            
            # Update predicted position
            current_pos = pred_trajectory[-1].copy()
            # Update all components (works for both 4DOF and 6DOF)
            for j in range(output_dim):
                current_pos[j] += delta_pred[j]
            pred_trajectory.append(current_pos)
            
            # Update ground truth position
            current_pos_true = true_trajectory[-1].copy()
            # Update all components (works for both 4DOF and 6DOF)
            for j in range(output_dim):
                current_pos_true[j] += delta_true[j]
            true_trajectory.append(current_pos_true)
            
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
    
    # Convert to numpy arrays
    true_trajectory = np.array(true_trajectory)
    pred_trajectory = np.array(pred_trajectory)
    
    # Calculate error metrics
    pos_error = np.sqrt(np.sum((true_trajectory[:, :3] - pred_trajectory[:, :3])**2, axis=1))
    
    # Handle angular errors differently based on dimensions
    if output_dim > 4:  # 6DOF case
        yaw_error = np.abs(np.mod(true_trajectory[:, 3] - pred_trajectory[:, 3] + np.pi, 2*np.pi) - np.pi)
        pitch_error = np.abs(np.mod(true_trajectory[:, 4] - pred_trajectory[:, 4] + np.pi, 2*np.pi) - np.pi)
        roll_error = np.abs(np.mod(true_trajectory[:, 5] - pred_trajectory[:, 5] + np.pi, 2*np.pi) - np.pi)
        mean_yaw_error = np.mean(yaw_error)
        mean_pitch_error = np.mean(pitch_error)
        mean_roll_error = np.mean(roll_error)
    else:  # 4DOF case
        yaw_error = np.abs(np.mod(true_trajectory[:, 3] - pred_trajectory[:, 3] + np.pi, 2*np.pi) - np.pi)
        mean_yaw_error = np.mean(yaw_error)
    
    mean_pos_error = np.mean(pos_error)
    
    # Create enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # XY plot (top view)
    axes[0, 0].plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', linewidth=2, label='Ground Truth')
    axes[0, 0].plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, label='Predicted')
    axes[0, 0].set_xlabel('X position (m)')
    axes[0, 0].set_ylabel('Y position (m)')
    axes[0, 0].set_title('Camera Trajectory - Top View (XY Plane)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # XZ plot (side view)
    axes[0, 1].plot(true_trajectory[:, 0], true_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
    axes[0, 1].plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, label='Predicted')
    axes[0, 1].set_xlabel('X position (m)')
    axes[0, 1].set_ylabel('Z position (m)')
    axes[0, 1].set_title('Camera Trajectory - Side View (XZ Plane)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Position error over time
    axes[1, 0].plot(pos_error, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Position error (m)')
    axes[1, 0].set_title(f'Position Error Over Time (Mean: {mean_pos_error:.4f} m)')
    axes[1, 0].grid(True)
    
    # Yaw error over time
    axes[1, 1].plot(yaw_error, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Yaw error (rad)')
    axes[1, 1].set_title(f'Yaw Error Over Time (Mean: {mean_yaw_error:.4f} rad)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Add overall title with camera and test info
    fig.suptitle('Trajectory Prediction Results', fontsize=16, y=1.02)
    
    if plot_save_path:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print(f"Saved trajectory plot to {plot_save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nTrajectory Prediction Results:")
    print(f"Total trajectory points: {len(true_trajectory)}")
    print(f"Mean position error: {mean_pos_error:.4f} m")
    print(f"Mean yaw error: {mean_yaw_error:.4f} rad")
    if output_dim > 4:  # 6DOF case
        print(f"Mean pitch error: {mean_pitch_error:.4f} rad")
        print(f"Mean roll error: {mean_roll_error:.4f} rad")
    print(f"Final position error: {pos_error[-1]:.4f} m")
    
    return true_trajectory, pred_trajectory, pos_error, yaw_error

def visualize_flow(flow, original_img=None, save_path=None):
    """
    Visualize optical flow using the HSV color wheel
    """
    # Convert flow to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # Set hue according to the angle of optical flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Set saturation to maximum
    hsv[..., 1] = 255
    
    # Set value according to the normalized magnitude of optical flow
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to BGR
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # If original image is provided, create a composite visualization
    if original_img is not None:
        # Ensure original image is in the correct format
        if original_img.dtype != np.uint8:
            # Convert from [-1, 1] range to [0, 255]
            original_img = ((original_img + 1) * 127.5).astype(np.uint8)
        
        # Resize if necessary
        if original_img.shape[:2] != flow.shape[:2]:
            original_img = cv2.resize(original_img, (flow.shape[1], flow.shape[0]))
        
        # Create a composite image (original on left, flow on right)
        composite = np.zeros((flow.shape[0], flow.shape[1] * 2, 3), dtype=np.uint8)
        composite[:, :flow.shape[1]] = original_img[:, :, :3]
        composite[:, flow.shape[1]:] = flow_rgb
        
        result_img = composite
    else:
        result_img = flow_rgb
    
    # Save if path is provided
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return result_img

def main():
    parser = argparse.ArgumentParser(description='Train FlowNet-to-Pose model for visual odometry')
    parser.add_argument('--data', required=True, help='Path to the image_pairs_with_gt.csv file')
    parser.add_argument('--root_dir', help='Root directory for image pairs (if not specified in CSV)')
    parser.add_argument('--model_dir', default='models_flownet2pose', help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--mode', choices=['train', 'test', 'predict', 'visualize_flow'], default='train', 
                        help='Mode: train, test, predict, or visualize_flow')
    parser.add_argument('--train_method', choices=['combined', 'two-stage'], default='two-stage',
                       help='Training method: combined (end-to-end) or two-stage')
    parser.add_argument('--model_path', help='Path to a saved model (for test and predict modes)')
    parser.add_argument('--plot_path', help='Path to save the trajectory plot')
    parser.add_argument('--train_cameras', default='0,1,2,3', help='Comma-separated list of camera IDs for training')
    parser.add_argument('--test_cameras', default='4', help='Comma-separated list of camera IDs for testing')
    parser.add_argument('--flow_vis_path', help='Path to save flow visualization (for visualize_flow mode)')
    parser.add_argument('--sample_image_pair', nargs=2, help='Paths to sample image pair for flow visualization')
    
    args = parser.parse_args()
    
    # Parse camera IDs
    train_cameras = [int(x) for x in args.train_cameras.split(',')]
    test_cameras = [int(x) for x in args.test_cameras.split(',')]
    
    if args.mode == 'train':
        print("Loading dataset by camera...")
        X_train, X_val, X_test, y_train, y_val, y_test, output_dim = load_dataset_by_camera(
            args.data, 
            train_cameras=train_cameras,
            test_cameras=test_cameras,
            root_dir=args.root_dir
        )
        
        print("\n=== Dataset Info ===")
        print(f"Output dimension: {output_dim} (4=XYZΨ, 6=XYZΨΘΦ)")
        print(f"X_train sample: {X_train[0]}")
        print(f"y_train sample: {y_train[0]}")
        
        # Determine flow and pose model input shapes
        # Flow model takes 6-channel input (stacked image pair)
        flow_input_shape = (256, 256, 6)
        # Flow output and pose input shape
        flow_output_shape = (64, 64, 2)  # Adjust based on your model architecture
        
        if args.train_method == 'two-stage':
            print("\nBuilding two-stage model (FlowNet -> Pose Regression)...")
            
            # Build FlowNet model
            flownet_model = build_flownet_model(input_shape=flow_input_shape)
            print("FlowNet model:")
            flownet_model.summary()
            
            # Build Pose Regression model
            pose_model = build_pose_regression_model(input_shape=flow_output_shape, output_dim=output_dim)
            print("\nPose Regression model:")
            pose_model.summary()
            
            # Train the models sequentially
            print("\nTraining two-stage model...")
            flownet_model, pose_model, combined_model = train_two_stage(
                flownet_model, pose_model, X_train, X_val, y_train, y_val,
                batch_size=args.batch_size, epochs=args.epochs,
                model_save_path=args.model_dir
            )
            
        else:  # 'combined'
            print("\nBuilding combined model (end-to-end)...")
            
            # Build FlowNet model
            flownet_model = build_flownet_model(input_shape=flow_input_shape)
            
            # Build Pose Regression model
            pose_model = build_pose_regression_model(input_shape=flow_output_shape, output_dim=output_dim)
            
            # Combine models
            combined_model = build_combined_model(flownet_model, pose_model, input_shape=flow_input_shape)
            print("Combined model:")
            combined_model.summary()
            
            # Train the combined model
            print("\nTraining combined model...")
            combined_model, history = train_combined_model(
                combined_model, X_train, X_val, y_train, y_val,
                batch_size=args.batch_size, epochs=args.epochs,
                model_save_path=args.model_dir
            )
        
        # Save test data for later
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        np.save(os.path.join(args.model_dir, 'test_data.npy'), test_data, allow_pickle=True)
        
        print(f"Model training complete. Models saved to {args.model_dir}")
        
        # Optional: immediately test on the holdout camera
        print("\nTesting on holdout camera data...")
        plot_path = args.plot_path or os.path.join(args.model_dir, 'camera_trajectory.png')
        true_traj, pred_traj, _, _ = predict_camera_trajectory(
            combined_model, X_test, y_test, 
            plot_save_path=plot_path
        )
        
    elif args.mode == 'test':
        if not args.model_path:
            parser.error("--model_path is required for test mode")
        
        print(f"Loading model from {args.model_path}...")
        combined_model = load_model(args.model_path)
        
        print("Loading test dataset...")
        try:
            # Try to load saved test data
            test_data = np.load(os.path.join(os.path.dirname(args.model_path), 'test_data.npy'), 
                              allow_pickle=True).item()
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            print(f"Loaded {len(X_test)} test samples from saved data")
        except:
            # Otherwise load from CSV
            _, _, X_test, _, _, y_test, _ = load_dataset_by_camera(
                args.data, 
                train_cameras=train_cameras,
                test_cameras=test_cameras,
                root_dir=args.root_dir
            )
        
        print("Evaluating model...")
        y_pred, mse, rmse, mae = evaluate_model(combined_model, X_test, y_test, batch_size=args.batch_size)
        
    elif args.mode == 'predict':
        if not args.model_path:
            parser.error("--model_path is required for predict mode")
        
        print(f"Loading model from {args.model_path}...")
        combined_model = load_model(args.model_path)
        
        print("Loading test dataset...")
        try:
            # Try to load saved test data
            test_data = np.load(os.path.join(os.path.dirname(args.model_path), 'test_data.npy'), 
                              allow_pickle=True).item()
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            print(f"Loaded {len(X_test)} test samples from saved data")
        except:
            # Otherwise load from CSV
            _, _, X_test, _, _, y_test, output_dim = load_dataset_by_camera(
                args.data, 
                train_cameras=train_cameras,
                test_cameras=test_cameras,
                root_dir=args.root_dir
            )
        
        # Set the starting position based on output dimension
        start_pos = np.zeros(y_test.shape[1])
        
        print("Predicting trajectory...")
        plot_path = args.plot_path or os.path.join(os.path.dirname(args.model_path), 'camera_trajectory.png')
        true_traj, pred_traj, pos_error, yaw_error = predict_camera_trajectory(
            combined_model, X_test, y_test, start_pos, plot_save_path=plot_path
        )
        
        print("Trajectory prediction complete.")
        
    elif args.mode == 'visualize_flow':
        if args.sample_image_pair:
            if len(args.sample_image_pair) != 2:
                parser.error("--sample_image_pair should specify two image paths")
            
            # Load the FlowNet model
            if args.model_path:
                print(f"Loading model from {args.model_path}...")
                # If the model is a combined model, get the flownet part
                combined_model = load_model(args.model_path)
                if hasattr(combined_model, 'get_layer') and combined_model.get_layer('flownet'):
                    flownet_model = Model(inputs=combined_model.input, 
                                          outputs=combined_model.get_layer('flownet').output)
                else:
                    # Assume it's just the flownet model
                    flownet_model = combined_model
            else:
                parser.error("--model_path is required for visualize_flow mode")
            
            # Load and preprocess the image pair
            img_pair = preprocess_image_pair(args.sample_image_pair[0], args.sample_image_pair[1])
            
            # Predict flow
            flow = flownet_model.predict(np.expand_dims(img_pair, axis=0))[0]
            
            # Convert the first image back to uint8 for visualization
            img_bgr = cv2.imread(args.sample_image_pair[0])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Visualize flow
            flow_vis = visualize_flow(flow, img_rgb, args.flow_vis_path)
            
            print(f"Flow visualization saved to {args.flow_vis_path}")
        else:
            parser.error("--sample_image_pair is required for visualize_flow mode")

if __name__ == "__main__":
    main()