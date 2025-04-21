import pandas as pd
import os
import argparse
import numpy as np
import re

def match_images_with_gt(image_pairs_dir, trajectory_csv, output_csv=None):
    """
    Match image pairs with ground truth trajectory data.
    
    Args:
        image_pairs_dir: Directory containing image pairs
        trajectory_csv: CSV file with trajectory data
        output_csv: Path to save matched data (optional)
        
    Returns:
        DataFrame with matched image pairs and trajectory data
    """
    print(f"Matching image pairs in {image_pairs_dir} with trajectory data from {trajectory_csv}")
    
    # Load trajectory data
    trajectory = pd.read_csv(trajectory_csv)
    print(f"Loaded {len(trajectory)} trajectory points")
    
    # Extract camera ID from directory name
    camera_id = None
    dir_name = os.path.basename(image_pairs_dir)
    if 'cam' in dir_name:
        match = re.search(r'cam(\d+)', dir_name)
        if match:
            camera_id = int(match.group(1))
            print(f"Detected camera ID: {camera_id}")
    
    # Get list of image pairs
    pairs = sorted([d for d in os.listdir(image_pairs_dir) if d.startswith('pair_')])
    print(f"Found {len(pairs)} image pairs")
    
    results = []
    
    for pair in pairs:
        pair_path = os.path.join(image_pairs_dir, pair)
        timestamp_file = os.path.join(pair_path, 'timestamp.txt')
        
        # Extract pair number from folder name
        pair_num = int(pair.split('_')[1])
        
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    prev_timestamp = float(lines[0].strip())
                    curr_timestamp = float(lines[1].strip())
                    mid_timestamp = (prev_timestamp + curr_timestamp) / 2
                else:
                    print(f"Warning: Invalid timestamp file format in {timestamp_file}")
                    continue
            
            # Find closest trajectory point
            idx = (trajectory['timestamp'] - mid_timestamp).abs().idxmin()
            gt_data = trajectory.iloc[idx]
            
            # Check if timestamp is too far from the closest point
            time_diff = abs(trajectory.iloc[idx]['timestamp'] - mid_timestamp)
            if time_diff > 0.1:  # 100 ms threshold
                print(f"Warning: Large time difference ({time_diff:.3f}s) for pair {pair}")
            
            results.append({
                'pair_id': pair,
                'pair_num': pair_num,
                'camera_id': camera_id,
                'prev_timestamp': prev_timestamp,
                'curr_timestamp': curr_timestamp,
                'trajectory_timestamp': gt_data['timestamp'],
                'delta_x': gt_data['delta_x'] if not np.isnan(gt_data['delta_x']) else 0.0,
                'delta_y': gt_data['delta_y'] if not np.isnan(gt_data['delta_y']) else 0.0,
                'delta_z': gt_data['delta_z'] if not np.isnan(gt_data['delta_z']) else 0.0,
                'delta_yaw': gt_data['delta_yaw'] if not np.isnan(gt_data['delta_yaw']) else 0.0,
                'delta_roll': gt_data['delta_roll'] if not np.isnan(gt_data['delta_roll']) else 0.0,
                'delta_pitch': gt_data['delta_pitch'] if not np.isnan(gt_data['delta_pitch']) else 0.0,
                'yaw': gt_data['yaw'],
                'pitch': gt_data['pitch'],
                'roll': gt_data['roll'],
                'image_pair_path': pair_path
            })
        else:
            print(f"Warning: No timestamp file found for {pair}")
    
    result_df = pd.DataFrame(results)
    
    # Print summary
    print(f"Successfully matched {len(result_df)} image pairs with trajectory data")
    if camera_id is not None:
        print(f"All data is from camera {camera_id}")
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"Saved matched data to {output_csv}")
    
    return result_df

def process_all_cameras(base_dir, trajectory_csv, output_dir):
    """
    Process image pairs from all camera directories and combine results
    
    Args:
        base_dir: Base directory containing camera folders
        trajectory_csv: Path to trajectory CSV file
        output_dir: Directory to save individual and combined CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all camera directories
    camera_dirs = [d for d in os.listdir(base_dir) if 'cam' in d and os.path.isdir(os.path.join(base_dir, d))]
    
    if not camera_dirs:
        print("No camera directories found. Expected directories containing 'cam' in the name.")
        return
    
    all_data = []
    
    # Process each camera directory
    for cam_dir in camera_dirs:
        cam_path = os.path.join(base_dir, cam_dir)
        print(f"\nProcessing camera directory: {cam_path}")
        
        # Individual output CSV for this camera
        cam_output = os.path.join(output_dir, f"{cam_dir}_gt.csv")
        
        # Match image pairs with ground truth data
        cam_data = match_images_with_gt(cam_path, trajectory_csv, cam_output)
        all_data.append(cam_data)
    
    # Combine all camera data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save combined dataset
    combined_output = os.path.join(output_dir, "all_cameras_with_gt.csv")
    combined_data.to_csv(combined_output, index=False)
    print(f"\nCombined data from all cameras saved to {combined_output}")
    print(f"Total image pairs: {len(combined_data)}")
    
    # Print camera statistics
    camera_counts = combined_data['camera_id'].value_counts().sort_index()
    print("\nImages per camera:")
    for camera, count in camera_counts.items():
        print(f"  Camera {camera}: {count} image pairs")
    
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Match image pairs with trajectory data')
    
    # Create mutually exclusive group for single vs all processing
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--process_all', action='store_true', help='Process all camera directories')
    group.add_argument('--pairs', help='Directory containing image pairs (for single camera processing)')
    
    # Common arguments
    parser.add_argument('--trajectory', required=True, help='CSV file with trajectory data')
    
    # Arguments for single camera mode
    parser.add_argument('--output', help='Path to save matched data CSV (for single camera)')
    
    # Arguments for process_all mode
    parser.add_argument('--base_dir', help='Base directory containing camera folders (for --process_all)')
    parser.add_argument('--output_dir', help='Directory to save outputs (for --process_all)')
    
    args = parser.parse_args()
    
    if args.process_all:
        if not args.base_dir:
            parser.error("--base_dir is required when using --process_all")
        if not args.output_dir:
            parser.error("--output_dir is required when using --process_all")
            
        combined_data = process_all_cameras(args.base_dir, args.trajectory, args.output_dir)
    else:
        if not args.pairs:
            parser.error("--pairs is required when not using --process_all")
        matched_data = match_images_with_gt(args.pairs, args.trajectory, args.output)
        print(matched_data.head())

if __name__ == "__main__":
    main()
#     main()