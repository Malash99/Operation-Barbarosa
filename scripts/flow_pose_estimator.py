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

import os
print("\n=== DEBUG START ===")
print(f"Current directory: {os.getcwd()}")
print(f"Files in /app/output: {os.listdir('/app/output')[:5]}...")

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU for training")
else:
    print("No GPU found. Using CPU for training")

def build_simplified_flownet(input_shape=(256, 256, 6), output_dim=4):
    """
    Build a simplified FlowNet model based on FlowNetS architecture
    """
    inputs = Input(shape=input_shape)
    
    # Contracting path (encoder)
    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (5, 5), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (5, 5), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    
    # Expanding path (decoder) for flow prediction
    up6 = UpSampling2D(size=(2, 2))(conv5)
    concat6 = Concatenate()([up6, conv4])
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(concat6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    concat7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu')(concat7)
    
    # Final flow prediction - output size will be 32x32
    flow = Conv2D(2, (3, 3), padding='same', activation='tanh', name='flow_output')(conv7)
    
    # Create the model with just flow output
    flow_model = Model(inputs=inputs, outputs=flow)
    
    # Add pose regression directly in this function
    # Extract features for pose prediction
    conv7a = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2d_7')(conv6)
    pool7 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4')(conv7a)
    conv8 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2d_8')(pool7)
    pool8 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_5')(conv8)
    
    # Flatten and add dense layers
    flat = Flatten()(pool8)
    fc1 = Dense(512, activation='relu')(flat)
    bn1 = BatchNormalization()(fc1)
    drop1 = Dropout(0.3)(bn1)
    fc2 = Dense(128, activation='relu')(drop1)
    bn2 = BatchNormalization()(fc2)
    drop2 = Dropout(0.3)(bn2)
    
    # Final regression output
    pose_output = Dense(output_dim, activation='linear', name='pose_output')(drop2)
    
    # Combined model with dual outputs
    combined_model = Model(inputs=inputs, outputs=[flow, pose_output])
    
    return combined_model

def build_pose_regression_model(flow_model, num_outputs=4):
    """
    Build a regression model that takes flow features and predicts pose changes
    """
    # Print all layer names to help debug
    for i, layer in enumerate(flow_model.layers):
        print(f"Layer {i}: {layer.name}")
    
    # Get the intermediate features from the flow model
    flow_features = flow_model.get_layer('conv2d_6').output
    
    # Add pose regression layers (already created in the model)
    # Use the existing pose_output from the model
    pose_output = flow_model.get_layer('pose_output').output
    
    # Create the combined model
    combined_model = Model(inputs=flow_model.input, outputs=[flow_model.output, pose_output])
    
    # Compile the model
    combined_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'flow_output': 'mse',
            'pose_output': 'mse'
        },
        loss_weights={
            'flow_output': 0.5,  # Weight for flow prediction loss
            'pose_output': 1.0   # Weight for pose prediction loss
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

def data_generator(image_pairs, targets, batch_size=16, target_size=(256, 256), shuffle=True):
    """
    Generator function to yield batches of data
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
            
            # For the combined model, we need to yield placeholder flow targets as well
            flow_shape = (X_batch.shape[0], target_size[0]//8, target_size[1]//8, 2)
            dummy_flow = np.zeros(flow_shape)
            
            yield X_batch, [dummy_flow, y_batch]

def train_model(model, X_train, X_val, y_train, y_val, batch_size=16, epochs=50, 
                model_save_path='models'):
    """
    Train the model
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create data generators
    train_gen = data_generator(X_train, y_train, batch_size=batch_size)
    val_gen = data_generator(X_val, y_val, batch_size=batch_size)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_save_path, 'flow_pose_model.h5'),
        monitor='val_pose_output_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_pose_output_loss',
        patience=10,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_pose_output_loss',
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
    model.save(os.path.join(model_save_path, 'flow_pose_model_final.h5'))
    
    # Save the training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_save_path, 'training_history.csv'), index=False)
    
    return model, history

def evaluate_model(model, X_test, y_test, batch_size=16):
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
        _, pose_pred = model.predict(X_batch)
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

def predict_camera_trajectory(model, X_test, y_test, start_pos=[0, 0, 0, 0], plot_save_path=None):
    """
    Predict trajectory for a specific camera and compare with ground truth
    """
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
            _, pose_pred = model.predict(stacked_imgs)
            delta_pred = pose_pred[0]
            
            # Get the true delta values
            delta_true = y_test[i]
            
            # Update predicted position
            current_pos = pred_trajectory[-1].copy()
            current_pos[0] += delta_pred[0]  # x
            current_pos[1] += delta_pred[1]  # y
            current_pos[2] += delta_pred[2]  # z
            current_pos[3] += delta_pred[3]  # yaw
            pred_trajectory.append(current_pos)
            
            # Update ground truth position
            current_pos_true = true_trajectory[-1].copy()
            current_pos_true[0] += delta_true[0]  # x
            current_pos_true[1] += delta_true[1]  # y
            current_pos_true[2] += delta_true[2]  # z
            current_pos_true[3] += delta_true[3]  # yaw
            true_trajectory.append(current_pos_true)
            
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
    
    # Convert to numpy arrays
    true_trajectory = np.array(true_trajectory)
    pred_trajectory = np.array(pred_trajectory)
    
    # Calculate error metrics
    pos_error = np.sqrt(np.sum((true_trajectory[:, :3] - pred_trajectory[:, :3])**2, axis=1))
    yaw_error = np.abs(true_trajectory[:, 3] - pred_trajectory[:, 3])
    
    mean_pos_error = np.mean(pos_error)
    mean_yaw_error = np.mean(yaw_error)
    
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
    
    # Add overall title
    fig.suptitle('Camera 4 Trajectory Prediction Results', fontsize=16, y=1.02)
    
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
    print(f"Final position error: {pos_error[-1]:.4f} m")
    
    return true_trajectory, pred_trajectory, pos_error, yaw_error


def main():
    parser = argparse.ArgumentParser(description='Train FlowNet-based pose estimator model')
    parser.add_argument('--data', required=True, help='Path to the image_pairs_with_gt.csv file')
    parser.add_argument('--root_dir', help='Root directory for image pairs (if not specified in CSV)')
    parser.add_argument('--model_dir', default='models', help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], default='train', 
                        help='Mode: train, test, or predict')
    parser.add_argument('--model_path', help='Path to a saved model (for test and predict modes)')
    parser.add_argument('--plot_path', help='Path to save the trajectory plot')
    parser.add_argument('--train_cameras', default='0,1,2,3', help='Comma-separated list of camera IDs for training')
    parser.add_argument('--test_cameras', default='4', help='Comma-separated list of camera IDs for testing')
    
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
        print("\n=== Debug: First 5 training samples ===")
        print(f"X_train[0]: {X_train[0]}")  # Should show image paths
        print(f"y_train[0]: {y_train[0]}")  # Should show [delta_x, delta_y, delta_z, delta_yaw]
        print("\n=== Data Shapes ===")
        print(f"X_train length: {len(X_train)}")
        print(f"y_train length: {len(y_train)}")
                
        print("Building model...")
        combined_model = build_simplified_flownet(output_dim=output_dim)
        
        # Compile the model
        combined_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                'flow_output': 'mse',
                'pose_output': 'mse'
            },
            loss_weights={
                'flow_output': 0.5,
                'pose_output': 1.0
            }
        )
        
        # Print model summary
        combined_model.summary()
        
        print("\n=== MODEL VERIFICATION ===")
        combined_model.summary()  # Use the correct variable name
        try:
            test_pred = combined_model.predict(np.zeros((1,256,256,6)))
            print("Test prediction succeeded:", test_pred)
        except Exception as e:
            print("MODEL PREDICTION FAILED:", str(e))
        print("Training model...")

        print("\n=== FINAL VERIFICATION ===")
        print(f"X_train length: {len(X_train)}")
        print(f"First image paths: {X_train[0]}")
        print(f"First label: {y_train[0]}")
        print("Attempting to load sample image...")
        sample_img = cv2.imread(X_train[0][0])
        print(f"Image loaded: {sample_img is not None}")

        combined_model, history = train_model(
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
        
        print(f"Model training complete. Model saved to {args.model_dir}")
        
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
        model = load_model(args.model_path)
        
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
            _, _, X_test, _, _, y_test = load_dataset_by_camera(
                args.data, 
                train_cameras=train_cameras,
                test_cameras=test_cameras,
                root_dir=args.root_dir
            )
        
        print("Evaluating model on camera 4...")
        y_pred, mse, rmse, mae = evaluate_model(model, X_test, y_test, batch_size=args.batch_size)
        
    elif args.mode == 'predict':
        if not args.model_path:
            parser.error("--model_path is required for predict mode")
        
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)
        
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
            _, _, X_test, _, _, y_test = load_dataset_by_camera(
                args.data, 
                train_cameras=train_cameras,
                test_cameras=test_cameras,
                root_dir=args.root_dir
            )
        
        # Assume the starting position is [0, 0, 0, 0] for [x, y, z, yaw]
        start_pos = [0, 0, 0, 0]
        
        print("Predicting camera 4 trajectory...")
        plot_path = args.plot_path or os.path.join(os.path.dirname(args.model_path), 'camera_trajectory.png')
        true_traj, pred_traj, pos_error, yaw_error = predict_camera_trajectory(
            model, X_test, y_test, start_pos, plot_save_path=plot_path
        )
        
        print("Trajectory prediction complete.")

if __name__ == "__main__": 
    main()