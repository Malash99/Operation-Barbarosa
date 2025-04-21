"""
FlowNet model implementation for optical flow estimation.
Part of Operation Barbarosa underwater visual odometry pipeline.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, applications

def create_flownet(input_shape=(256, 256, 6)):
    """
    Implementation of a simplified FlowNet architecture to estimate optical flow
    from consecutive image pairs.
    
    Args:
        input_shape: Input shape (H, W, 6) - stacked image pairs (3 channels each)
        
    Returns:
        FlowNet model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Use MobileNetV2 as base model (lighter than traditional FlowNet)
    # Remove the last 1000-class layer and use the feature extractor
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # No pretrained weights for 6-channel input
    )
    
    # Encoder (feature extraction)
    x = base_model(inputs)
    
    # Decoder (upsampling to generate flow)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    
    # Final output: 2-channel flow (x and y directions)
    flow = layers.Conv2D(2, kernel_size=3, padding='same', activation=None, name='flow')(x)
    
    # Global average pooling to get a fixed-size representation
    flow_features = layers.GlobalAveragePooling2D()(flow)
    
    model = Model(inputs=inputs, outputs=flow_features)
    return model


def create_simple_flownet(input_shape=(256, 256, 6)):
    """
    Implementation of a lightweight FlowNet for optical flow estimation.
    This version is simpler and more efficient for training on CPU.
    
    Args:
        input_shape: Input shape (H, W, 6) - stacked image pairs (3 channels each)
        
    Returns:
        FlowNet model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Additional feature extraction
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Flow prediction
    flow = layers.Conv2D(2, kernel_size=3, padding='same', activation=None, name='flow')(x)
    
    # Global average pooling to get fixed-size representation (2D to 1D)
    flow_features = layers.GlobalAveragePooling2D()(flow)
    
    # Add some dense layers to extract useful features from the flow
    x = layers.Dense(128, activation='relu')(flow_features)
    flow_embedding = layers.Dense(64, activation='relu', name='flow_embedding')(x)
    
    model = Model(inputs=inputs, outputs=flow_embedding)
    return model