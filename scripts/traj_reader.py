import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import argparse
import os

def process_tum_file(file_path, output_path=None):
    """
    Process a TUM format trajectory file and convert quaternions to Euler angles.
    
    Args:
        file_path: Path to the TUM format file
        output_path: Path to save the processed CSV file (optional)
        
    Returns:
        DataFrame with processed trajectory data
    """
    print(f"Processing trajectory file: {file_path}")
    
    # Read TUM format file (timestamp, x, y, z, qx, qy, qz, qw)
    data = pd.read_csv(file_path, sep=' ', header=None, 
                      names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    
    print(f"Loaded {len(data)} trajectory points")
    
    # Convert quaternions to Euler angles
    euler_angles = []
    for _, row in data.iterrows():
        q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
        yaw, pitch, roll = q.yaw_pitch_roll
        euler_angles.append([yaw, pitch, roll])
    
    euler_df = pd.DataFrame(euler_angles, columns=['yaw', 'pitch', 'roll'])
    result = pd.concat([data, euler_df], axis=1)
    
    # Calculate deltas between consecutive poses
    result['delta_x'] = result['x'].diff()
    result['delta_y'] = result['y'].diff()
    result['delta_z'] = result['z'].diff()
    result['delta_yaw'] = result['yaw'].diff()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved processed trajectory to {output_path}")
    
    return result

def plot_trajectory(trajectory_data, output_path=None):
    """
    Plot the trajectory path.
    
    Args:
        trajectory_data: DataFrame with processed trajectory data
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(trajectory_data['x'], trajectory_data['y'])
    plt.title('Trajectory Path')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved trajectory plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process TUM format trajectory file')
    parser.add_argument('--input', required=True, help='Path to TUM format trajectory file')
    parser.add_argument('--output', help='Path to save processed CSV file')
    parser.add_argument('--plot', help='Path to save trajectory plot')
    
    args = parser.parse_args()
    
    trajectory_data = process_tum_file(args.input, args.output)
    print(trajectory_data.head())
    
    if args.plot:
        plot_trajectory(trajectory_data, args.plot)

if __name__ == "__main__":
    main()