this is the trial using deepseek after the failure of sonnet to extract the image pairs

///////////////////////////////////////////////////////////////////////////////////////
Building and Running the Container

docker build -t ros-tensorflow .
docker run -it --rm -v "C:\Users\Admin\Desktop\research\second trial\:/app" ros-tensorflow

///////////////////////////////////////////////////////////////////////////////////////
run the scripts then 

python3 scripts/traj_reader.py



python3 scripts/image_pair.py
python3 scripts/pairs_with_trajectory.py
python3 scripts/visualization.py

////////

permanent fix for the ocker files 
RUN pip3 install \
    numpy==1.21.0 \
    pandas==1.3.0 \
    matplotlib==3.4.3 \
    tensorflow==2.6.0 \
    pyquaternion \
    opencv-python==4.5.3.56

docker build -t ros-tensorflow .