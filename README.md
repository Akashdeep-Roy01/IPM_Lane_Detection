# IPM_Lane_Detection

A simple ROS2 package to detect lanes from images by using Inverse Perspective Mapping.

To use:

1. Build the docker file using `docker build -t <image name> .`
2. Change the image name inside "docker_run.sh" and run it using `bash docker_run.sh`
3. Build the package using `colcon build --symlink-install`
4. Source using `source /docker_ws/install/setup.bash`
5. Run the nodes `image_publisher.py` and `ipm_lane_detect.py`

Results
Input

https://github.com/Akashdeep-Roy01/IPM_Lane_Detection/assets/99131809/d3b6b420-b444-4612-bc72-cf1d9ba14f98

Output

https://github.com/Akashdeep-Roy01/IPM_Lane_Detection/assets/99131809/210411b9-d4f4-4374-af49-8faef88231bd







