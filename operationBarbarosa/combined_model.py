"""
Combined FlowNet and Pose Estimator model for underwater visual odometry.
Part of Operation Barbarosa pipeline.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from flownet_model import create_simple_flownet
from pose_estimator import create_pose_network, weighted_mse_loss, sincos_loss

class DataNormalizer:
    """
    Class for normalizing and denormalizing pose data
    """
    def __init__(self, method='none'):
        """
        Initialize normalizer with specified method
        
        Args:
            method: Normalization method ('none', 'minmax', 'meanstd')
        """
        self.method = method
        self.translation_mean = None
        self.translation_std = None
        self.rotation_mean = None
        self.rotation_std = None
        self.translation_min = None
        self.translation_max = None
        self.rotation_min = None
        self.rotation_max = None
        
    def fit(self, pose_data):
        """
        Compute normalization parameters from data
        
        Args:
            pose_data: Array of pose data [dx, dy, dz, dyaw, droll, dpitch]
        """
        if self.method == 'none':
            return
            
        translation = pose_data[:, 0:3]
        rotation = pose_data[:, 3:6]
        
        if self.method == 'meanstd':
            # Compute mean and standard deviation
            self.translation_mean = np.mean(translation, axis=0)
            self.translation_std = np.std(translation, axis=0)
            # Add small epsilon to avoid division by zero
            self.translation_std = np.maximum(self.translation_std, 1e-5)
            
            self.rotation_mean = np.mean(rotation, axis=0)
            self.rotation_std = np.std(rotation, axis=0)
            self.rotation_std = np.maximum(self.rotation_std, 1e-5)
            
        elif self.method == 'minmax':
            # Compute min and max values
            self.translation_min = np.min(translation, axis=0)
            self.translation_max = np.max(translation, axis=0)
            # Ensure there's a range to normalize
            range_check = self.translation_max - self.translation_min
            self.translation_min = np.where(range_check < 1e-5, 
                                           self.translation_min - 0.5, 
                                           self.translation_min)
            self.translation_max = np.where(range_check < 1e-5, 
                                           self.translation_max + 0.5, 
                                           self.translation_max)
            
            self.rotation_min = np.min(rotation, axis=0)
            self.rotation_max = np.max(rotation, axis=0)
            range_check = self.rotation_max - self.rotation_min
            self.rotation_min = np.where(range_check < 1e-5, 
                                        self.rotation_min - 0.5, 
                                        self.rotation_min)
            self.rotation_max = np.where(range_check < 1e-5, 
                                        self.rotation_max + 0.5, 
                                        self.rotation_max)
    
    def normalize(self, pose_data):
        """
        Normalize pose data
        
        Args:
            pose_data: Array of pose data [dx, dy, dz, dyaw, droll, dpitch]
            
        Returns:
            Normalized pose data
        """
        if self.method == 'none':
            return pose_data
            
        normalized_data = np.copy(pose_data)
        
        if self.method == 'meanstd':
            # Normalize translation using mean and std
            normalized_data[:, 0:3] = (pose_data[:, 0:3] - self.translation_mean) / self.translation_std
            # Normalize rotation using mean and std
            normalized_data[:, 3:6] = (pose_data[:, 3:6] - self.rotation_mean) / self.rotation_std
            
        elif self.method == 'minmax':
            # Normalize translation to [0, 1] range
            t_range = self.translation_max - self.translation_min
            normalized_data[:, 0:3] = (pose_data[:, 0:3] - self.translation_min) / t_range
            
            # Normalize rotation to [0, 1] range
            r_range = self.rotation_max - self.rotation_min
            normalized_data[:, 3:6] = (pose_data[:, 3:6] - self.rotation_min) / r_range
            
        return normalized_data
        
    def denormalize(self, normalized_data):
        """
        Denormalize pose data
        
        Args:
            normalized_data: Normalized pose data
            
        Returns:
            Original scale pose data
        """
        if self.method == 'none':
            return normalized_data
            
        denormalized_data = np.copy(normalized_data)
        
        if self.method == 'meanstd':
            # Denormalize translation
            denormalized_data[:, 0:3] = (normalized_data[:, 0:3] * self.translation_std) + self.translation_mean
            # Denormalize rotation
            denormalized_data[:, 3:6] = (normalized_data[:, 3:6] * self.rotation_std) + self.rotation_mean
            
        elif self.method == 'minmax':
            # Denormalize translation from [0, 1] range
            t_range = self.translation_max - self.translation_min
            denormalized_data[:, 0:3] = (normalized_data[:, 0:3] * t_range) + self.translation_min
            
            # Denormalize rotation from [0, 1] range
            r_range = self.rotation_max - self.rotation_min
            denormalized_data[:, 3:6] = (normalized_data[:, 3:6] * r_range) + self.rotation_min
            
        return denormalized_data
        
    def save(self, filepath):
        """
        Save normalizer parameters to file
        
        Args:
            filepath: Path to save file
        """
        save_dict = {
            'method': self.method,
            'translation_mean': self.translation_mean,
            'translation_std': self.translation_std,
            'rotation_mean': self.rotation_mean,
            'rotation_std': self.rotation_std,
            'translation_min': self.translation_min,
            'translation_max': self.translation_max,
            'rotation_min': self.rotation_min,
            'rotation_max': self.rotation_max
        }
        np.save(filepath, save_dict, allow_pickle=True)
        
    def load(self, filepath):
        """
        Load normalizer parameters from file
        
        Args:
            filepath: Path to load file
        """
        load_dict = np.load(filepath, allow_pickle=True).item()
        self.method = load_dict['method']
        self.translation_mean = load_dict['translation_mean']
        self.translation_std = load_dict['translation_std']
        self.rotation_mean = load_dict['rotation_mean']
        self.rotation_std = load_dict['rotation_std']
        self.translation_min = load_dict['translation_min']
        self.translation_max = load_dict['translation_max']
        self.rotation_min = load_dict['rotation_min']
        self.rotation_max = load_dict['rotation_max']


def create_combined_model(input_shape=(256, 256, 6), use_weighted_loss=True, use_sincos_loss=True):
    """
    Create a combined FlowNet + Pose Estimator model
    
    Args:
        input_shape: Input shape for image pairs
        use_weighted_loss: Whether to use weighted MSE loss
        use_sincos_loss: Whether to use sin/cos loss for rotation
        
    Returns:
        Combined model
    """
    # Create FlowNet model
    flownet = create_simple_flownet(input_shape=input_shape)
    
    # Get flow features output shape
    flow_output_shape = flownet.output.shape[1:]
    
    # Create pose estimator network
    pose_net = create_pose_network(input_dim=flow_output_shape[0])
    
    # Create combined model
    image_input = layers.Input(shape=input_shape)
    flow_features = flownet(image_input)
    pose_output = pose_net(flow_features)
    
    combined_model = Model(inputs=image_input, outputs=pose_output)
    
    # Compile model with appropriate loss function
    if use_sincos_loss:
        combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=sincos_loss
        )
    elif use_weighted_loss:
        combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=weighted_mse_loss
        )
    else:
        combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
    
    return combined_model, flownet, pose_net


def train_flownet_separately(flow_inputs, flow_targets, epochs=30, batch_size=8):
    """
    Train FlowNet separately for optical flow estimation
    
    Args:
        flow_inputs: Image pair inputs
        flow_targets: Optical flow targets
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Trained FlowNet model
    """
    # Create FlowNet model
    input_shape = flow_inputs.shape[1:]
    flownet = create_simple_flownet(input_shape=input_shape)
    
    # Compile model
    flownet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train model
    flownet.fit(
        flow_inputs,
        flow_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    
    return flownet