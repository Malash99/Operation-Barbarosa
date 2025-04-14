import pandas as pd
import os
from datetime import datetime

def match_images_with_gt(image_pairs_dir, trajectory_csv):
    # Load trajectory data
    trajectory = pd.read_csv(trajectory_csv)
    
    # Get list of image pairs
    pairs = sorted([d for d in os.listdir(image_pairs_dir) if d.startswith('pair_')])
    
    results = []
    
    for pair in pairs:
        pair_path = os.path.join(image_pairs_dir, pair)
        timestamp_file = os.path.join(pair_path, 'timestamp.txt')
        
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                timestamp = float(f.read().strip())
            
            # Find closest trajectory point
            idx = (trajectory['timestamp'] - timestamp).abs().idxmin()
            gt_data = trajectory.iloc[idx]
            
            results.append({
                'pair_id': pair,
                'timestamp': timestamp,
                'delta_x': gt_data['delta_x'],
                'delta_y': gt_data['delta_y'],
                'delta_z': gt_data['delta_z'],
                'delta_yaw': gt_data['delta_yaw'],
                'image_pair_path': pair_path
            })
    
    return pd.DataFrame(results)

# Usage (after running previous scripts)
matched_data = match_images_with_gt('image_pairs', 'processed_trajectory.csv')
matched_data.to_csv('image_pairs_with_gt.csv', index=False)