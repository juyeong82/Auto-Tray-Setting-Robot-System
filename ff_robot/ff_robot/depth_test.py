#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class QuickCheck(Node):
    def __init__(self):
        super().__init__('quick_check')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, 
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.callback, 
            10
        )
        self.count = 0
        
    def callback(self, msg):
        self.count += 1
        if self.count % 10 != 0:  # 10프레임마다 출력
            return
            
        depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        h, w = depth.shape
        
        # 중앙 5x5 영역 평균
        cx, cy = w//2, h//2
        roi = depth[cy-2:cy+3, cx-2:cx+3]
        valid = roi[roi > 0]
        
        if len(valid) > 0:
            avg = np.mean(valid)
            print(f"📏 중앙 깊이: {avg:.0f}mm ({avg/10:.1f}cm)")
        else:
            print("⚠️ Depth = 0 (측정 실패)")

rclpy.init()
node = QuickCheck()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()