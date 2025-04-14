import rosbag
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

def extract_image_pairs(bag_file, output_dir, topic='/camera/image_raw'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bridge = CvBridge()
    prev_image = None
    prev_timestamp = None
    pair_count = 0
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                if prev_image is not None:
                    # Save the pair
                    pair_dir = os.path.join(output_dir, f'pair_{pair_count:04d}')
                    os.makedirs(pair_dir, exist_ok=True)
                    
                    cv2.imwrite(os.path.join(pair_dir, 'prev.png'), prev_image)
                    cv2.imwrite(os.path.join(pair_dir, 'current.png'), cv_image)
                    
                    # You could add visualization of the difference here
                    diff = cv2.absdiff(prev_image, cv_image)
                    cv2.imwrite(os.path.join(pair_dir, 'diff.png'), diff)
                    
                    pair_count += 1
                
                prev_image = cv_image
                prev_timestamp = t.to_sec()
                
            except Exception as e:
                print(f"Error processing image: {e}")
    
    print(f"Extracted {pair_count} image pairs")

# Usage
extract_image_pairs('ariel_2023-12-21-14-26-32_2.bag', 'image_pairs')

print(f"Total messages in bag: {bag.get_message_count()}")
print(f"Messages on image topic: {bag.get_message_count(topic_filter='/camera/image_raw')}")