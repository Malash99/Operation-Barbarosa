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
# Use tensorflow-cpu instead of tensorflow to avoid GPU dependencies
RUN pip3 install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib \
    tensorflow-cpu==2.10.0 \
    scikit-learn \
    tqdm \
    pyquaternion

# Create directories for project
RUN mkdir -p /app/data /app/output /app/scripts

# Copy your code into the container
COPY scripts/ /app/scripts/

# Set the working directory
WORKDIR /app

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
export CUDA_VISIBLE_DEVICES=""\n\
exec "$@"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]