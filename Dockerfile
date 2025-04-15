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
RUN pip3 install \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib \
    tensorflow==2.10.0 \
    scikit-learn \
    tqdm \
    pyquaternion

# Create directories for project
RUN mkdir -p /app/data /app/output

# Copy your code into the container
COPY scripts/ /app/scripts/

# Set the working directory
WORKDIR /app

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
exec "$@"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]