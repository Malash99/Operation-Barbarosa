import cv2
import pandas as pd
import matplotlib.pyplot as plt

def visualize_pair(pair_data):
    prev_img = cv2.imread(os.path.join(pair_data['image_pair_path'], 'prev.png'))
    curr_img = cv2.imread(os.path.join(pair_data['image_pair_path'], 'current.png'))
    
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.imshow(prev_img)
    ax1.set_title('Previous Frame')
    ax1.axis('off')
    
    ax2.imshow(curr_img)
    ax2.set_title(f"Current Frame\nΔx: {pair_data['delta_x']:.3f}, Δy: {pair_data['delta_y']:.3f}\nΔθ: {pair_data['delta_yaw']:.3f}")
    ax2.axis('off')
    
    plt.show()

# Load matched data
matched_data = pd.read_csv('image_pairs_with_gt.csv')

# Visualize first few pairs
for _, row in matched_data.head(3).iterrows():
    visualize_pair(row)