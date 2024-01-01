# IPM_Lane_Detection

![ROS2](https://img.shields.io/badge/ROS2-Humble-%23F46800.svg?style=for-the-badge&logo=ROS2-Humble&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

A simple ROS2 package to detect lanes from images by using Inverse Perspective Mapping.

To use:

1. Build the docker file using `docker build -t <image name> .`
2. Change the image name inside "docker_run.sh" and run it using `bash docker_run.sh`
3. Build the package using `colcon build --symlink-install`
4. Source using `source /docker_ws/install/setup.bash`
5. Run the nodes `image_publisher.py` and `ipm_lane_detect.py`

Results

Input

https://github.com/Akashdeep-Roy01/IPM_Lane_Detection/assets/99131809/be4267ed-0e72-4af0-aff9-1f8bfcdbaca8

Output

https://github.com/Akashdeep-Roy01/IPM_Lane_Detection/assets/99131809/210411b9-d4f4-4374-af49-8faef88231bd










