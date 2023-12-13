#!/usr/bin/env python3
import cv2 
import rclpy
from rclpy.node import Node 
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image
from ipm_utilities import *
 
class IPM_LanePublisher(Node):
  def __init__(self):
    super().__init__('image_subscriber')
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 
      10)
    self.subscription 
    self.br = CvBridge()
    self.publisher_ = self.create_publisher(Image, 'ipm_frames', 10)

  def listener_callback(self, data):
    self.get_logger().info('Receiving video frame')
    current_frame = self.br.imgmsg_to_cv2(data)
    lane_lines_img,_,_,_,_,_ = lane_detect_ipm_pipeline(current_frame)
    self.publisher_.publish(self.br.cv2_to_imgmsg(lane_lines_img))
    # self.publisher_.publish(data)
    # cv2.imshow("camera", current_frame)
    # cv2.waitKey(1)
  
def main(args=None):
  
  rclpy.init(args=args)
  ipm_publisher = IPM_LanePublisher()
  rclpy.spin(ipm_publisher)
#   ipm_publisher.destroy_node()
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()