"""
Pose estimator model implementation for Operation Barbarosa.
Takes optical flow features and predicts 6 DOF pose changes.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import math

def create_pose_network(input_dim=64):
    """
    Create a network that predicts 6 DOF pose changes from optical flow features
    
    Args:
        input_dim: Dimension of the input flow features
        
    Returns:
        Pose estimation model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Several dense layers for pose regression
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Translation branch (x, y, z)
    translation = layers.Dense(3, activation=None, name='translation')(x)
    
    # Rotation branch (yaw, roll, pitch)
    rotation = layers.Dense(3, activation=None, name='rotation')(x)
    
    # Combine outputs
    outputs = layers.Concatenate(name='pose_output')([translation, rotation])
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def sincos_loss(y_true, y_pred):
    """
    Custom loss function for angle prediction that handles periodicity.
    Converts angles to sin/cos representation before computing loss.
    
    Args:
        y_true: Ground truth angles (in radians)
        y_pred: Predicted angles (in radians)
        
    Returns:
        Loss value
    """
    # Extract rotation components (last 3 elements)
    rot_true = y_true[:, 3:6]
    rot_pred = y_pred[:, 3:6]
    
    # Convert to sin/cos representation
    sin_true = tf.sin(rot_true)
    cos_true = tf.cos(rot_true)
    sin_pred = tf.sin(rot_pred)
    cos_pred = tf.cos(rot_pred)
    
    # Compute MSE on sin/cos values
    sin_loss = tf.reduce_mean(tf.square(sin_true - sin_pred))
    cos_loss = tf.reduce_mean(tf.square(cos_true - cos_pred))
    
    # Extract translation components (first 3 elements)
    trans_true = y_true[:, 0:3]
    trans_pred = y_pred[:, 0:3]
    
    # Regular MSE for translation
    trans_loss = tf.reduce_mean(tf.square(trans_true - trans_pred))
    
    # Combine losses, with angle loss potentially weighted higher
    angle_weight = 2.0  # Weight for angle loss
    return trans_loss + angle_weight * (sin_loss + cos_loss)


def weighted_mse_loss(y_true, y_pred):
    """
    Weighted MSE loss that gives more importance to rotation components.
    
    Args:
        y_true: Ground truth values [dx, dy, dz, dyaw, droll, dpitch]
        y_pred: Predicted values [dx, dy, dz, dyaw, droll, dpitch]
        
    Returns:
        Weighted MSE loss
    """
    # Define weights for each component
    # Higher weights for rotation components to address small angle changes
    weights = tf.constant([1.0, 1.0, 1.0, 5.0, 5.0, 5.0], dtype=tf.float32)
    
    # Compute squared errors
    squared_errors = tf.square(y_true - y_pred)
    
    # Apply weights
    weighted_errors = squared_errors * weights
    
    # Return mean error
    return tf.reduce_mean(weighted_errors)