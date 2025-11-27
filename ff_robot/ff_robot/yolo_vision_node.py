#!/usr/bin/env python3
# yolo_vision_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time

# 서비스 인터페이스
from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point, Quaternion

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        # [경로 확인 필수]
        self.model_path = "/home/yang_seungwon/ros2_ws/src/ff_robot/models/best.pt" 
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.camera_intrinsics = None 
        
        self.get_logger().info(f"YOLO 모델 로딩 중... ({self.model_path})")
        try:
            self.model = YOLO(self.model_path) 
            self.get_logger().info("✅ YOLO11 모델 로드 완료")
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")

        # RealSense Topic
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)

        self.srv = self.create_service(DetectObject, '/dsr01/detect_object', self.handle_detect_object)
        self.get_logger().info("👀 Real YOLO Vision Node (Eye-in-Hand Mode) Ready")

    def color_callback(self, msg):
        try: self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def depth_callback(self, msg):
        try: self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except: pass

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            K = msg.k
            self.camera_intrinsics = {'fx': K[0], 'fy': K[4], 'ppx': K[2], 'ppy': K[5]}

    def pixel_to_3d_cam(self, u, v, depth_mm):
        """ 2D 픽셀 -> 3D 카메라 좌표계 (Xc, Yc, Zc) """
        if self.camera_intrinsics is None: return None
        z = depth_mm / 1000.0 # meter 단위
        if z <= 0: return None
        
        x = (u - self.camera_intrinsics['ppx']) * z / self.camera_intrinsics['fx']
        y = (v - self.camera_intrinsics['ppy']) * z / self.camera_intrinsics['fy']
        return np.array([x, y, z])

    def handle_detect_object(self, request, response):
        target_name = request.target_object_id
        self.get_logger().info(f"🔎 YOLO 요청: '{target_name}'")

        if self.latest_color_img is None or self.latest_depth_img is None:
            response.found = False
            return response

        results = self.model(self.latest_color_img, verbose=False)
        found_target = False
        center_x, center_y = 0, 0

        for r in results:
            for box in r.obb: # OBB 사용
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                if cls_name == target_name:
                    c_x, c_y, w, h, rot = box.xywhr[0].cpu().numpy()
                    center_x, center_y = int(c_x), int(c_y)
                    found_target = True
                    break
            if found_target: break

        if not found_target:
            self.get_logger().warn(f"   ⚠️ '{target_name}' 못 찾음")
            response.found = False
            return response

        # Depth 추출 (중심점 주변 Median 필터 적용)
        roi_size = 5
        h, w = self.latest_depth_img.shape
        x_min = max(0, center_x - roi_size//2)
        x_max = min(w, center_x + roi_size//2)
        y_min = max(0, center_y - roi_size//2)
        y_max = min(h, center_y + roi_size//2)
        
        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) == 0:
            response.found = False
            return response
            
        depth_mm = np.median(valid_depths)

        # 3D 변환 (Camera Frame)
        cam_point = self.pixel_to_3d_cam(center_x, center_y, depth_mm)
        if cam_point is None:
            response.found = False
            return response

        # [중요] 여기서는 로봇 베이스 변환을 하지 않고, 카메라 기준 좌표만 보냅니다.
        # 변환은 Controller가 현재 팔 각도를 고려해서 수행합니다.
        response.found = True
        response.position = Point(x=cam_point[0], y=cam_point[1], z=cam_point[2])
        response.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) 
        
        self.get_logger().info(f"   📸 Camera Coords: X={cam_point[0]:.3f}, Y={cam_point[1]:.3f}, Z={cam_point[2]:.3f}")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()