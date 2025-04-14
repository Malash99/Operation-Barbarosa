import pandas as pd
import os
import argparse
import numpy as np

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
    
    # Get list of image pairs
    pairs = sorted([d for d in os.listdir(image_pairs_dir) if d.startswith('pair_')])
    print(f"Found {len(pairs)} image pairs")
    
    results = []
    
    for pair in pairs:
        pair_path = os.path.join(image_pairs_dir, pair)
        timestamp_file = os.path.join(pair_path, 'timestamp.txt')
        
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
                'prev_timestamp': prev_timestamp,
                'curr_timestamp': curr_timestamp,
                'trajectory_timestamp': gt_data['timestamp'],
                'delta_x': gt_data['delta_x'] if not np.isnan(gt_data['delta_x']) else 0.0,
                'delta_y': gt_data['delta_y'] if not np.isnan(gt_data['delta_y']) else 0.0,
                'delta_z': gt_data['delta_z'] if not np.isnan(gt_data['delta_z']) else 0.0,
                'delta_yaw': gt_data['delta_yaw'] if not np.isnan(gt_data['delta_yaw']) else 0.0,
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
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"Saved matched data to {output_csv}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Match image pairs with trajectory data')
    parser.add_argument('--pairs', required=True, help='Directory containing image pairs')
    parser.add_argument('--trajectory', required=True, help='CSV file with trajectory data')
    parser.add_argument('--output', help='Path to save matched data CSV')
    
    args = parser.parse_args()
    
    matched_data = match_images_with_gt(args.pairs, args.trajectory, args.output)
    print(matched_data.head())

if __name__ == "__main__":
    main()