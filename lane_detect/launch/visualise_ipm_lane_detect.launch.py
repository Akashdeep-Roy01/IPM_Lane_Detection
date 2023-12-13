import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_path = os.path.join(get_package_share_directory('lane_detect'))
    default_rviz_config_path = os.path.join(pkg_path, 'config','rviz_config.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )
    # To use the single launch file - create a folder /docker_ws/install/lane_detect/lib/lane_detect and then copy the executables into it
    lane_publisher = Node(
        package="lane_detect",
        executable="image_publisher.py",
        output="screen")
    
    ipm_lane_publisher = Node(
        package="lane_detect",
        executable="ipm_lane_detect.py",
        output="screen")
    
    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value='True',description='Use sim time if true'),
        DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
                                            description='Absolute path to rviz config file'),
        lane_publisher,
        ipm_lane_publisher,
        rviz_node,
    ])