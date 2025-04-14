import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

def process_tum_file(file_path):
    # Read TUM format file (timestamp, x, y, z, qx, qy, qz, qw)
    data = pd.read_csv(file_path, sep=' ', header=None, 
                      names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    
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
    
    return result

# Usage
trajectory_data = process_tum_file('qualisys_ariel_odom_traj_8_id6.tum')
trajectory_data.to_csv('processed_trajectory.csv', index=False)
print(trajectory_data.head())

# Plot trajectory
plt.figure(figsize=(12, 6))
plt.plot(trajectory_data['x'], trajectory_data['y'])
plt.title('Trajectory Path')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()