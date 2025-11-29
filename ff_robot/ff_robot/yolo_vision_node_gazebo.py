#!/usr/bin/env python3
# yolo_vision_node_gazebo.py - Gazebo 토픽 매칭 버전

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point, Quaternion

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        self.get_logger().info("=========================================")
        self.get_logger().info("🚀 YOLO Vision Node (Gazebo Matched)")
        self.get_logger().info("=========================================")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "models", "best.pt")
        
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.camera_intrinsics = None 
        
        try:
            self.model = YOLO(self.model_path)
            device = self.model.device
            self.get_logger().info(f"✅ YOLO 모델 로드 완료 (Device: {device})")
            
            # GPU 워밍업
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            self.get_logger().info("✅ GPU 워밍업 완료")
            
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")

        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cb_group = ReentrantCallbackGroup()

        # ✅ Gazebo 토픽으로 변경
        self.create_subscription(
            Image, '/dsr01/camera/color/image_raw',  # Gazebo RGB
            self.color_callback, qos_profile, callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/dsr01/camera/depth/image_rect_raw',  # Gazebo Depth
            self.depth_callback, qos_profile, callback_group=self.cb_group
        )
        self.create_subscription(
            CameraInfo, '/dsr01/d435i/camera_info',  # Gazebo Camera Info
            self.info_callback, qos_profile, callback_group=self.cb_group
        )

        self.srv = self.create_service(
            DetectObject, '/dsr01/detect_object', 
            self.handle_detect_object, callback_group=self.cb_group
        )
        self.get_logger().info("👀 서비스 서버 준비 완료")

    def color_callback(self, msg):
        try: 
            self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB 변환 실패: {e}")

    def depth_callback(self, msg):
        try: 
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Depth 변환 실패: {e}")

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            K = msg.k
            self.camera_intrinsics = {'fx': K[0], 'fy': K[4], 'ppx': K[2], 'ppy': K[5]}
            self.get_logger().info(f"✅ 카메라 파라미터: fx={K[0]:.1f}, fy={K[4]:.1f}, ppx={K[2]:.1f}, ppy={K[5]:.1f}")

    def pixel_to_3d_cam(self, u, v, depth_mm):
        if self.camera_intrinsics is None: 
            return None
        z = depth_mm / 1000.0 
        if z <= 0: 
            return None
        x = (u - self.camera_intrinsics['ppx']) * z / self.camera_intrinsics['fx']
        y = (v - self.camera_intrinsics['ppy']) * z / self.camera_intrinsics['fy']
        return np.array([x, y, z])

    def handle_detect_object(self, request, response):
        target_name = request.target_object_id
        self.get_logger().info(f"🔎 YOLO 요청 수신: '{target_name}'")

        inference_start = time.time()

        # 이미지 수신 확인
        if self.latest_color_img is None or self.latest_depth_img is None:
            wait_start = time.time()
            while self.latest_color_img is None or self.latest_depth_img is None:
                if time.time() - wait_start > 1.0:
                    self.get_logger().error(f"❌ [Timeout] 이미지 수신 실패")
                    response.found = False
                    return response
                time.sleep(0.01)

        # YOLO 추론
        yolo_start = time.time()
        results = self.model(self.latest_color_img, verbose=False)
        yolo_time = (time.time() - yolo_start) * 1000
        
        found_target = False
        center_x, center_y = 0, 0
        box_w, box_h = 20, 20
        rotation_rad = 0.0

        for r in results:
            if r.obb is not None:
                for box in r.obb:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name == target_name:
                        c_x, c_y, w, h, rot = box.xywhr[0].cpu().numpy()
                        center_x, center_y = int(c_x), int(c_y)
                        box_w, box_h = int(w), int(h)
                        rotation_rad = float(rot)
                        found_target = True
                        self.get_logger().info(f"OBB 회전각: {np.degrees(rotation_rad):.1f}°")
                        break
            if not found_target and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name == target_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = int((x1+x2)/2)
                        center_y = int((y1+y2)/2)
                        box_w, box_h = abs(x2-x1), abs(y2-y1)
                        found_target = True
                        break

        if not found_target:
            self.get_logger().warn(f"   ⚠️ 타겟 '{target_name}' 없음")
            response.found = False
            return response

        # Depth 추출 (ROI)
        h, w = self.latest_depth_img.shape
        roi_w = max(5, int(box_w * 0.3))
        roi_h = max(5, int(box_h * 0.3))
        x_min = max(0, center_x - roi_w // 2)
        x_max = min(w, center_x + roi_w // 2)
        y_min = max(0, center_y - roi_h // 2)
        y_max = min(h, center_y + roi_h // 2)
        
        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) == 0:
            depth_mm = 600.0
            self.get_logger().warn(f"   ⚠️ Depth 0 -> 고정값 600mm 사용")
        else:
            depth_mm = float(np.median(valid_depths))
            self.get_logger().info(f"   📏 거리: {depth_mm:.1f} mm")

        # 3D 변환
        cam_point = self.pixel_to_3d_cam(center_x, center_y, depth_mm)
        if cam_point is None:
            self.get_logger().error("   ❌ 3D 변환 실패")
            response.found = False
            return response

        response.found = True
        response.position = Point(x=cam_point[0], y=cam_point[1], z=cam_point[2])
        
        # OBB 회전각을 Quaternion으로 변환
        half_angle = rotation_rad / 2.0
        qz = np.sin(half_angle)
        qw = np.cos(half_angle)
        response.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        
        # mm로 변환 후 오프셋 적용
        x_mm = cam_point[0] * 1000.0
        y_mm = cam_point[1] * 1000.0 - 20.0
        z_mm = cam_point[2] * 1000.0 + 170.0
        
        total_time = (time.time() - inference_start) * 1000
        self.get_logger().info(
            f"   ✅ 보정 좌표(mm): X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} | 회전: {np.degrees(rotation_rad):.1f}°"
            f" | YOLO: {yolo_time:.0f}ms, 전체: {total_time:.0f}ms"
        )
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: 
        executor.spin()
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
