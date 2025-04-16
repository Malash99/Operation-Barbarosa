@echo off
REM Flow-to-Pose Visual Odometry Pipeline Runner

echo ==========================================================
echo FlowNet to Pose Visual Odometry Pipeline
echo ==========================================================

REM Check if an argument was provided
set MODE=%1
if "%MODE%"=="" set MODE=train

REM Set the second argument as train method (two-stage or combined)
set TRAIN_METHOD=%2
if "%TRAIN_METHOD%"=="" set TRAIN_METHOD=two-stage

REM Set the third argument as train cameras
set TRAIN_CAMERAS=%3
if "%TRAIN_CAMERAS%"=="" set TRAIN_CAMERAS=0,1,2,3

REM Set the fourth argument as test cameras
set TEST_CAMERAS=%4
if "%TEST_CAMERAS%"=="" set TEST_CAMERAS=4

echo Running in %MODE% mode with %TRAIN_METHOD% method...
echo Training on cameras: %TRAIN_CAMERAS%
echo Testing on cameras: %TEST_CAMERAS%

REM Run the container
docker run -it --rm ^
  -v "%cd%\data:/app/data" ^
  -v "%cd%\output:/app/output" ^
  -v "%cd%\scripts:/app/scripts" ^
  ros-trajectory-analysis bash -c "^
    echo 'Checking for TensorFlow...' && ^
    python3 -c 'import tensorflow as tf; print(\"TensorFlow version:\", tf.__version__)' && ^
    ^
    if [ '%MODE%' = 'train' ]; then ^
      echo 'Training FlowNet to Pose model...' && ^
      python3 /app/scripts/flownet_2_pose_estimator.py --data /app/output/all_cameras_with_gt.csv --root_dir /app/output --model_dir /app/output/models_flownet2pose --mode train --train_method %TRAIN_METHOD% --batch_size 8 --epochs 50 --train_cameras %TRAIN_CAMERAS% --test_cameras %TEST_CAMERAS%; ^
    elif [ '%MODE%' = 'test' ]; then ^
      echo 'Testing model performance...' && ^
      python3 /app/scripts/flownet_2_pose_estimator.py --data /app/output/all_cameras_with_gt.csv --root_dir /app/output --mode test --model_path /app/output/models_flownet2pose/combined_model_final.h5 --test_cameras %TEST_CAMERAS%; ^
    elif [ '%MODE%' = 'predict' ]; then ^
      echo 'Predicting trajectory...' && ^
      python3 /app/scripts/flownet_2_pose_estimator.py --data /app/output/all_cameras_with_gt.csv --root_dir /app/output --mode predict --model_path /app/output/models_flownet2pose/combined_model_final.h5 --plot_path /app/output/flownet2pose_trajectory.png --test_cameras %TEST_CAMERAS%; ^
    elif [ '%MODE%' = 'flow' ]; then ^
      echo 'Visualizing optical flow...' && ^
      python3 /app/scripts/flownet_2_pose_estimator.py --mode visualize_flow --model_path /app/output/models_flownet2pose/flownet_model_final.h5 --sample_image_pair /app/output/image_pairs_cam%TEST_CAMERAS%/pair_0000/prev.png /app/output/image_pairs_cam%TEST_CAMERAS%/pair_0000/current.png --flow_vis_path /app/output/flow_visualization.png; ^
    else ^
      echo 'Invalid mode specified. Use train, test, predict, or flow.'; ^
    fi ^
  "

echo ==========================================================
echo Process complete!
echo ==========================================================

if "%MODE%"=="predict" (
  echo Check the output directory for the trajectory plot: output\flownet2pose_trajectory.png
)
if "%MODE%"=="flow" (
  echo Check the output directory for the flow visualization: output\flow_visualization.png
)
if "%MODE%"=="train" (
  echo Trained models saved to: output\models_flownet2pose\
)