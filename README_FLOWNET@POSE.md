# FlowNet to Pose Estimator

This implementation uses a two-stage approach for visual odometry:

1. First stage: FlowNet model generates optical flow from consecutive image pairs
2. Second stage: Pose regression network uses the flow field to predict 6DOF motion (position and orientation changes)

## Architecture Comparison

### Original Approach (flow_pose_estimator.py)
- **Single Network**: Uses a shared encoder for both flow and pose prediction
- **Flow as Auxiliary**: Flow prediction is a side task that helps with feature learning
- **End-to-End**: Trained in a single pass
- **View-Dependent**: Features may be more camera-specific

### New Approach (flownet_2_pose_estimator.py)
- **Two-Stage Pipeline**: Explicit flow computation followed by pose regression
- **Flow as Intermediate**: Optical flow provides a more direct motion representation
- **Training Options**: Can be trained end-to-end or in stages
- **View-Invariant**: Optical flow is more robust to viewpoint changes
- **Interpretable**: Flow visualizations provide insight into the model's understanding

## Key Features

- Pre-computes explicit optical flow fields that can be visualized
- Can train in two ways: end-to-end or two-stage approach
- Custom loss function that handles the circular nature of angular measurements
- Support for both 4DOF (x, y, z, yaw) and 6DOF (x, y, z, yaw, pitch, roll) motion
- Enhanced visualization tools including optical flow visualization

## Usage

### Training

```bash
# Train using two-stage approach
./run_flownet_2_pose.bat train two-stage 0,1,2,3 4

# Train using end-to-end approach
./run_flownet_2_pose.bat train combined 0,1,2,3 4

# Train on camera 0 only, test on camera 1
./run_flownet_2_pose.bat train two-stage 0 1
```

### Testing

```bash
# Test the model on camera 4
./run_flownet_2_pose.bat test two-stage 0,1,2,3 4
```

### Trajectory Prediction

```bash
# Predict and visualize trajectory for camera 4
./run_flownet_2_pose.bat predict two-stage 0,1,2,3 4
```

### Flow Visualization

```bash
# Visualize optical flow for a sample image pair from camera 4
./run_flownet_2_pose.bat flow two-stage 0,1,2,3 4
```

## Advanced Usage

You can run the Python script directly with more options:

```bash
python3 /app/scripts/flownet_2_pose_estimator.py --data /app/output/all_cameras_with_gt.csv --root_dir /app/output --model_dir /app/output/models_flownet2pose --mode train --train_method two-stage --batch_size 8 --epochs 50 --train_cameras 0 --test_cameras 1
```

## Model Details

### FlowNet Architecture
- Input: 6-channel image (RGB pair stacked)
- Output: 2-channel optical flow field (x and y displacements)
- Architecture: Based on FlowNetS with convolutional encoder-decoder

### Pose Regression Architecture
- Input: Optical flow field from FlowNet
- Output: 4 or 6 motion parameters (delta_x, delta_y, delta_z, delta_yaw, [delta_pitch, delta_roll])
- Architecture: Convolutional layers followed by dense regression

## Improving Results

- **Pre-training**: Use a pre-trained FlowNet model for better initial flow estimates
- **Data Augmentation**: Apply random rotations, flips, and brightness changes to improve robustness
- **Recurrent Structure**: Add LSTM/GRU layers to incorporate temporal information
- **Camera Parameters**: Incorporate intrinsic camera parameters to better handle different cameras
- **Scale Normalization**: Normalize motion parameters to similar ranges for better training

## Known Limitations

1. Pure visual odometry suffers from scale ambiguity
2. Performance degrades in areas with limited visual features
3. Rotation estimation is typically harder than translation
4. Accumulating errors can cause drift over long trajectories