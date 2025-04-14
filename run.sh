#!/bin/bash
# Script to run the complete data processing pipeline

set -e  # Exit on error

echo "============================================="
echo "ROS Bag and Trajectory Analysis Pipeline"
echo "============================================="

# Create necessary directories
mkdir -p data output/image_pairs output/plots

# Check if data files exist
if [ ! -f "data/ariel_2023-12-21-14-26-32_2.bag" ]; then
  echo "Error: ROS bag file not found in data directory."
  echo "Please place 'ariel_2023-12-21-14-26-32_2.bag' in the data directory."
  exit 1
fi

if [ ! -f "data/qualisys_ariel_odom_traj_8_id6.tum" ]; then
  echo "Error: TUM trajectory file not found in data directory."
  echo "Please place 'qualisys_ariel_odom_traj_8_id6.tum' in the data directory."
  exit 1
fi

# Step 1: Process trajectory data
echo "Step 1: Processing trajectory data..."
python3 scripts/traj_reader.py \
  --input data/qualisys_ariel_odom_traj_8_id6.tum \
  --output output/processed_trajectory.csv \
  --plot output/plots/trajectory_plot.png

# Step 2: Extract image pairs from ROS bag
echo "Step 2: Extracting image pairs from ROS bag..."
python3 scripts/image_pair.py \
  --bag data/ariel_2023-12-21-14-26-32_2.bag \
  --output output/image_pairs \
  --topic "${ROS_TOPIC:-/camera/image_raw}" \
  --time_diff "${IMAGE_FREQ:-0.1}"

# Step 3: Match image pairs with trajectory data
echo "Step 3: Matching image pairs with trajectory data..."
python3 scripts/pairs_with_trajectory.py \
  --pairs output/image_pairs \
  --trajectory output/processed_trajectory.csv \
  --output output/image_pairs_with_gt.csv

# Step 4: Visualize image pairs
echo "Step 4: Visualizing image pairs..."
python3 scripts/visualization.py \
  --input output/image_pairs_with_gt.csv \
  --output_dir output/visualizations \
  --num_pairs 2 \
  --plot_trajectory \
  --trajectory_plot output/plots/deltas_plot.png

echo "============================================="
echo "Processing complete!"
echo "Results saved in the output directory"
echo "============================================="