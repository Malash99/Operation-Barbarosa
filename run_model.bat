@echo off
REM Flow-based Visual Odometry Training and Prediction Script

echo ==========================================================
echo Flow-based Visual Odometry Pipeline
echo ==========================================================

REM Check if an argument was provided
set MODE=%1
if "%MODE%"=="" set MODE=train

echo Running in %MODE% mode...

REM Run the container with GPU support if available
docker run -it --rm -v "%cd%\data:/app/data" -v "%cd%\output:/app/output" -v "%cd%\scripts:/app/scripts" ros-trajectory-analysis bash -c "echo 'Checking for GPU...' && python3 -c 'import tensorflow as tf; print(\"GPU available:\", len(tf.config.list_physical_devices(\"GPU\")) > 0)' && if [ '%MODE%' = 'train' ]; then echo 'Training FlowNet pose estimation model...' && python3 /app/scripts/flow_pose_estimator.py --data /app/output/image_pairs_with_gt.csv --root_dir /app/output/image_pairs --model_dir /app/output/models --mode train --batch_size 8 --epochs 50; elif [ '%MODE%' = 'test' ]; then echo 'Testing model performance...' && python3 /app/scripts/flow_pose_estimator.py --data /app/output/image_pairs_with_gt.csv --root_dir /app/output/image_pairs --mode test --model_path /app/output/models/flow_pose_model.h5; elif [ '%MODE%' = 'predict' ]; then echo 'Predicting trajectory...' && python3 /app/scripts/flow_pose_estimator.py --data /app/output/image_pairs_with_gt.csv --root_dir /app/output/image_pairs --mode predict --model_path /app/output/models/flow_pose_model.h5 --plot_path /app/output/trajectory_plot.png; else echo 'Invalid mode specified. Use train, test, or predict.'; fi"

echo ==========================================================
echo Process complete!
echo ==========================================================

if "%MODE%"=="predict" (
  echo Check the output directory for the trajectory plot: output\trajectory_plot.png
)
if "%MODE%"=="train" (
  echo Trained model saved to: output\models\flow_pose_model.h5
)