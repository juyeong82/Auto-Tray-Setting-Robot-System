#!/usr/bin/env python3
"""
Gazebo 카메라 뷰어
RGB, Depth 이미지를 OpenCV 창으로 표시합니다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        self.bridge = CvBridge()
        
        # RGB 이미지 구독
        self.rgb_sub = self.create_subscription(
            Image,
            '/dsr01/camera/color/image_raw',
            self.rgb_callback,
            10
        )
        
        # Depth 이미지 구독
        self.depth_sub = self.create_subscription(
            Image,
            '/dsr01/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # Camera Info 구독
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/dsr01/d435i/camera_info',
            self.info_callback,
            10
        )
        
        self.camera_info = None
        self.frame_count = 0
        
        self.get_logger().info("📷 카메라 뷰어 시작!")
        self.get_logger().info("   RGB 토픽: /dsr01/camera/color/image_raw")
        self.get_logger().info("   Depth 토픽: /dsr01/camera/depth/image_rect_raw")
        self.get_logger().info("👉 'q' 키를 눌러 종료하세요")
        
    def info_callback(self, msg):
        """카메라 정보 수신"""
        if self.camera_info is None:
            self.camera_info = msg
            K = msg.k
            self.get_logger().info(f"\n📐 카메라 파라미터:")
            self.get_logger().info(f"   해상도: {msg.width} x {msg.height}")
            self.get_logger().info(f"   fx: {K[0]:.1f}, fy: {K[4]:.1f}")
            self.get_logger().info(f"   cx: {K[2]:.1f}, cy: {K[5]:.1f}")
        
    def rgb_callback(self, msg):
        """RGB 이미지 처리"""
        try:
            # ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 프레임 카운트 증가
            self.frame_count += 1
            
            # 정보 오버레이
            info_text = f"Frame: {self.frame_count} | Size: {cv_image.shape[1]}x{cv_image.shape[0]}"
            cv2.putText(cv_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 중심 십자선 표시
            h, w = cv_image.shape[:2]
            cv2.line(cv_image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
            cv2.line(cv_image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)
            
            # 표시
            cv2.imshow('RGB Camera', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"RGB 처리 에러: {e}")
            
    def depth_callback(self, msg):
        """Depth 이미지 처리"""
        try:
            # ROS Image → OpenCV (16-bit)
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            
            # Depth를 시각화용으로 변환 (0-255)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            # 중심점의 깊이 값 표시
            h, w = depth_image.shape
            center_depth = depth_image[h//2, w//2]
            depth_text = f"Center Depth: {center_depth} mm ({center_depth/1000:.2f} m)"
            cv2.putText(depth_colormap, depth_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 중심 십자선
            cv2.line(depth_colormap, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 2)
            cv2.line(depth_colormap, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 2)
            
            # 표시
            cv2.imshow('Depth Camera', depth_colormap)
            
            # 'q' 키로 종료
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.get_logger().info("종료 중...")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"Depth 처리 에러: {e}")

def main():
    rclpy.init()
    viewer = CameraViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        viewer.get_logger().info("\n👋 종료")
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
