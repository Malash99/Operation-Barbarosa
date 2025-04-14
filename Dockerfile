# Use ROS Noetic as base image
FROM osrf/ros:noetic-desktop-full

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-tf \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Replace the existing pip install line with:
    RUN pip3 install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib \
    tensorflow \
    pyquaternion \
    opencv-python

# Create workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws/src

# Copy your code into the container
COPY . /app
WORKDIR /app

CMD ["bash"]