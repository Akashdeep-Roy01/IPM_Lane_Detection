# IPM_Lane_Detection

![ROS2](https://camo.githubusercontent.com/b874e7cbc7323284002070083cf5fc1cfff41a3a5573f598638564fa4017fd65/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f524f5320322d68756d626c652d626c756576696f6c6574)
![Docker](https://camo.githubusercontent.com/b609225bdb4ad668a23ec18b022f166bd86e184b28ddcf34b604f4a68837907f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f636b65722d3234393645443f7374796c653d666c61742d737175617265266c6f676f3d646f636b6572266c6f676f436f6c6f723d7768697465)
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

https://github.com/Akashdeep-Roy01/IPM_Lane_Detection/assets/99131809/d3b6b420-b444-4612-bc72-cf1d9ba14f98

Output

https://github.com/Akashdeep-Roy01/IPM_Lane_Detection/assets/99131809/210411b9-d4f4-4374-af49-8faef88231bd







