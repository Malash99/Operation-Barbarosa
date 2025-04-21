# Operation Barbarosa: Underwater Visual Odometry

This README provides a comprehensive guide to setting up and running Operation Barbarosa, a deep learning-based visual odometry system designed for underwater environments.

## Overview

Operation Barbarosa processes consecutive underwater camera images to predict 6-DOF pose changes (translations and rotations) using a two-stage neural network approach:

1. **Stage 1 (FlowNet)**: Estimates optical flow between consecutive images
2. **Stage 2 (Pose Network)**: Predicts 6-DOF pose changes from optical flow features

## Prerequisites

- ROS Noetic
- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- pandas, numpy, matplotlib
- pyquaternion

## Directory Structure

```
/app/
├── data/                      # Input data directory
│   ├── ariel_2023-12-21-14-26-32_2.bag  # ROS bag with camera images
│   └── qualisys_ariel_odom_traj_8_id6.tum  # Ground truth trajectory
│
├── output/                    # Output directory (created during execution)
│
└── operationBarbarosa/        # Python scripts
    ├── flownet_model.py       # FlowNet implementation
    ├── pose_estimator.py      # Pose estimation network
    ├── combined_model.py      # Combined model pipeline
    ├── train.py               # Training script
    ├── predict.py             # Prediction and visualization
    ├── data_utils.py          # Data processing utilities
    └── run_scripts.sh         # Helper scripts
```

## Setup

1. Ensure your Docker environment is set up correctly:

```Dockerfile
FROM osrf/ros:noetic-desktop-full

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-tf \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib \
    tensorflow-cpu==2.10.0 \
    scikit-learn \
    tqdm \
    pyquaternion

# Create directories
RUN mkdir -p /app/data /app/output /app/operationBarbarosa

# Set the working directory
WORKDIR /app

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
export CUDA_VISIBLE_DEVICES=""\n\
exec "$@"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

2. Copy all the Python scripts to the `/app/operationBarbarosa/` directory.

3. Make the run scripts executable:
```bash
chmod +x /app/operationBarbarosa/run_scripts.sh
```

## Step-by-Step Guide

### 1. Process Trajectory Data

Convert ground truth trajectory data from TUM format to a CSV file:

```bash
python3 /app/operationBarbarosa/traj_reader.py \
  --input /app/data/qualisys_ariel_odom_traj_8_id6.tum \
  --output /app/output/processed_trajectory.csv \
  --plot /app/output/plots/trajectory_plot.png
```

### 2. Extract Image Pairs from ROS Bags

Extract consecutive image pairs from ROS bags for each camera (0-4):

```bash
# For camera 0
python3 /app/operationBarbarosa/image_pair.py \
  --bag /app/data/ariel_2023-12-21-14-26-32_2.bag \
  --output /app/output/image_pairs_cam0 \
  --topic "/cam0" \
  --topic "/alphasense_driver_ros/cam0" \
  --time_diff 0.1

# Repeat for cameras 1-4, changing camera numbers accordingly
```

### 3. Match Image Pairs with Trajectory Data

Associate image pairs with ground truth pose changes:

```bash
python3 /app/operationBarbarosa/pairs_with_trajectory.py \
  --process_all \
  --base_dir /app/output \
  --trajectory /app/output/processed_trajectory.csv \
  --output_dir /app/output
```

### 4. Analyze Dataset (Optional)

Analyze the dataset statistics:

```bash
python3 /app/operationBarbarosa/data_utils.py \
  --analyze \
  --data /app/output/all_cameras_with_gt.csv \
  --output_dir /app/output/analysis
```

### 5. Train the FlowNet to Pose Model

Train the combined model for pose estimation:

```bash
python3 /app/operationBarbarosa/train.py \
  --data /app/output/all_cameras_with_gt.csv \
  --model_dir /app/output/models_flownet2pose_meanstd \
  --batch_size 8 \
  --epochs 50 \
  --train_cameras 0,1,2,3 \
  --test_cameras 4 \
  --normalize_method meanstd \
  --loss_type sincos
```

Training options:
- **normalize_method**: 
  - `none`: No normalization
  - `minmax`: Min-max scaling to [0,1]
  - `meanstd`: Z-score normalization (recommended)
- **loss_type**:
  - `mse`: Standard Mean Squared Error
  - `weighted`: Weighted MSE (higher weights for rotations)
  - `sincos`: Sine-cosine loss for rotations (recommended for angles)

### 6. Run Trajectory Prediction

Predict trajectories for all cameras:

```bash
python3 /app/operationBarbarosa/predict.py \
  --model_path /app/output/models_flownet2pose_meanstd/combined_model_final.h5 \
  --data_dir /app/output \
  --output_dir /app/output/camera_results_meanstd \
  --cameras 0,1,2,3,4 \
  --normalize_method meanstd
```

## Results

### Camera 0 Results

![Camera 0 Trajectory Comparison](camera0_trajectory.png)

The trajectory comparison for Camera 0 shows:
- Initial tracking follows the ground truth path
- Drift increases after approximately 200 trajectory points
- Scale and orientation errors accumulate over time
- Final position error of approximately 23 meters

### Camera 4 Results

![Camera 4 Trajectory Comparison](camera4_trajectory.png)

The trajectory comparison for Camera 4 shows:
- Significant Z-axis drift compared to Camera 0
- More pronounced deviation from ground truth
- Larger cumulative error throughout the trajectory
- Final position error of approximately 40 meters

## Analysis and Improvements

Based on the results, several improvements could be made:

1. **Enhanced Loss Function**:
```python
def enhanced_sincos_loss(y_true, y_pred):
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
    
    # Scale-aware translation loss
    trans_magnitude = tf.maximum(tf.reduce_sum(tf.square(trans_true), axis=1), 1e-6)
    trans_errors = tf.square(trans_true - trans_pred)
    trans_loss = tf.reduce_mean(trans_errors / tf.expand_dims(tf.sqrt(trans_magnitude), axis=1))
    
    # Higher weight for angular components
    angle_weight = 8.0  # Increased from 2.0
    
    # Combined angle loss
    angle_loss = (sin_loss + cos_loss) * angle_weight
    
    return trans_loss + angle_loss
```

2. **Increase Training Duration**: Ensure model trains for full 50 epochs

3. **Cross-Camera Training**: Include some data from Camera 4 in training set

4. **Post-processing Filtering**: Implement Kalman or particle filtering

5. **Scale Recovery**: Add techniques to correct scale drift

## Important Implementation Notes

### File Naming

The system expects image files to be named:
- `prev.png`: Previous image in the pair
- `current.png`: Current image in the pair

### Data Generator

The training data generator is designed to repeat infinitely for training across multiple epochs. This is implemented with a `while True:` loop in the generator function.

### Handling Missing or Corrupt Images

The system includes error handling for missing or corrupt images, replacing them with zero tensors to avoid training interruptions.

### Trajectory Calculation

Trajectory calculation is performed using quaternions to accurately represent rotations in 3D space. The system keeps track of cumulative poses to reconstruct the full trajectory.

## Evaluation Metrics

The system evaluates performance using:
- **Mean Position Error**: Average Euclidean distance between predicted and ground truth positions
- **Mean Orientation Error**: Average angular difference between predicted and ground truth orientations
- **Final Position Error**: Position error at the end of the trajectory
- **Maximum Errors**: Maximum position and orientation errors across the trajectory

## Troubleshooting

### 3D Plot Issues

If you encounter errors with 3D plotting:

```
ValueError: Unknown projection '3d'
```

You may need to install additional matplotlib components:

```bash
apt-get update && apt-get install -y python3-matplotlib
```

Or modify the `plot_trajectories` function in `predict.py` to use only 2D plots.

### Image Loading Errors

If you see errors like:

```
OpenCV(4.2.0) ../modules/imgproc/src/resize.cpp:4045: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
```

Check that:
1. The image files exist and are valid
2. The file naming is consistent (`prev.png` and `current.png`)
3. The paths in the CSV file are correct

## Running the Complete Pipeline

For convenience, you can run the entire pipeline with:

```bash
bash /app/operationBarbarosa/run_scripts.sh full
```

Or run specific steps:

```bash
bash /app/operationBarbarosa/run_scripts.sh step 5  # Run only the training step
```

## Next Steps

Potential enhancements to the system:
- Integration of IMU data for more robust odometry
- Loop closure detection for drift correction
- Online calibration for multi-camera setups
- Scale recovery methods for monocular setups
