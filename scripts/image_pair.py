import rosbag
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import argparse

def extract_image_pairs(bag_file, output_dir, topic='/camera/image_raw', min_time_diff=0.1):
    """
    Extract consecutive image pairs from a ROS bag file.
    
    Args:
        bag_file: Path to the ROS bag file
        output_dir: Directory to save the image pairs
        topic: ROS topic containing the images
        min_time_diff: Minimum time difference between frames (seconds)
    """
    print(f"Extracting image pairs from {bag_file}")
    print(f"Topic: {topic}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bridge = CvBridge()
    prev_image = None
    prev_timestamp = None
    pair_count = 0
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            total_messages = bag.get_message_count()
            topic_messages = bag.get_message_count(topic_filters=[topic])
            
            print(f"Total messages in bag: {total_messages}")
            print(f"Messages on image topic: {topic_messages}")
            
            for topic, msg, t in bag.read_messages(topics=[topic]):
                try:
                    curr_timestamp = t.to_sec()
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    
                    if prev_image is not None:
                        # Only save if enough time has passed
                        time_diff = curr_timestamp - prev_timestamp
                        if time_diff >= min_time_diff:
                            # Save the pair
                            pair_dir = os.path.join(output_dir, f'pair_{pair_count:04d}')
                            os.makedirs(pair_dir, exist_ok=True)
                            
                            cv2.imwrite(os.path.join(pair_dir, 'prev.png'), prev_image)
                            cv2.imwrite(os.path.join(pair_dir, 'current.png'), cv_image)
                            
                            # Save timestamps
                            with open(os.path.join(pair_dir, 'timestamp.txt'), 'w') as f:
                                f.write(f"{prev_timestamp}\n{curr_timestamp}")
                            
                            # You could add visualization of the difference here
                            diff = cv2.absdiff(prev_image, cv_image)
                            cv2.imwrite(os.path.join(pair_dir, 'diff.png'), diff)
                            
                            # Print progress every 10 pairs
                            if pair_count % 10 == 0:
                                print(f"Extracted {pair_count} image pairs")
                            
                            pair_count += 1
                            
                            # Update previous image (skip some frames)
                            prev_image = cv_image.copy()
                            prev_timestamp = curr_timestamp
                    else:
                        # First image
                        prev_image = cv_image.copy()
                        prev_timestamp = curr_timestamp
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
    
    except Exception as e:
        print(f"Error opening bag file: {e}")
    
    print(f"Extracted a total of {pair_count} image pairs")
    return pair_count

def main():
    parser = argparse.ArgumentParser(description='Extract image pairs from ROS bag')
    parser.add_argument('--bag', required=True, help='Path to ROS bag file')
    parser.add_argument('--output', required=True, help='Directory to save image pairs')
    parser.add_argument('--topic', default='/camera/image_raw', help='Image topic')
    parser.add_argument('--time_diff', type=float, default=0.1, help='Minimum time between frames')
    
    args = parser.parse_args()
    
    extract_image_pairs(args.bag, args.output, args.topic, args.time_diff)

if __name__ == "__main__":
    main()