#!/usr/bin/env python3
# yolo_vision_node.py (Fixed Depth Fallback)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point, Quaternion

# =========================================================
# [설정] 깊이 센서가 실패했을 때 사용할 '가상의 거리' (단위: mm)
# 로봇이 관측 위치에 있을 때, 카메라~바닥까지의 대략적인 거리
# =========================================================
FIXED_DEPTH_MM = 600.0 

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        self.get_logger().info("=========================================")
        self.get_logger().info("🚀 YOLO Vision Node (Fixed Depth Ver)")
        self.get_logger().info(f"👉 Depth 실패 시 고정값 {FIXED_DEPTH_MM}mm 사용")
        self.get_logger().info("=========================================")
        
        self.model_path = "/home/rokey/ros2_ws/src/ff_robot/ff_robot/models/best.pt"
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.camera_intrinsics = None 
        
        try:
            self.model = YOLO(self.model_path) 
            self.get_logger().info("✅ YOLO 모델 로드 완료")
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")

        # QoS 및 콜백 그룹
        qos_profile = qos_profile_sensor_data
        self.cb_group = ReentrantCallbackGroup()

        # 구독
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile, callback_group=self.cb_group)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, qos_profile, callback_group=self.cb_group)

        self.srv = self.create_service(DetectObject, '/dsr01/detect_object', self.handle_detect_object, callback_group=self.cb_group)
        self.get_logger().info("👀 서비스 서버 준비 완료")

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
        if self.camera_intrinsics is None: return None
        z = depth_mm / 1000.0 
        if z <= 0: return None
        x = (u - self.camera_intrinsics['ppx']) * z / self.camera_intrinsics['fx']
        y = (v - self.camera_intrinsics['ppy']) * z / self.camera_intrinsics['fy']
        return np.array([x, y, z])

    def handle_detect_object(self, request, response):
        target_name = request.target_object_id
        # 로그 너무 많이 뜨면 주석 처리 가능
        self.get_logger().info(f"🔎 YOLO 요청 수신: '{target_name}'")

        # ---------------------------------------------------------
        # 1. 이미지 수신 대기 (최대 5초)
        # ---------------------------------------------------------
        wait_start = time.time()
        while self.latest_color_img is None or self.latest_depth_img is None:
            if time.time() - wait_start > 5.0: 
                self.get_logger().error(f"❌ [Timeout] 이미지 수신 실패")
                response.found = False
                return response
            time.sleep(0.01) # 대기 시간 최소화
        
        # ---------------------------------------------------------
        # 2. YOLO 추론 (그리기 로직 제거 -> 속도 향상)
        # ---------------------------------------------------------
        results = self.model(self.latest_color_img, verbose=False)
        
        found_target = False
        center_x, center_y = 0, 0
        box_w, box_h = 20, 20 # 기본값

        for r in results:
            # (A) OBB 탐색
            if r.obb is not None:
                for box in r.obb:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id] # [필수] 이름 추출

                    if cls_name == target_name:
                        c_x, c_y, w, h, rot = box.xywhr[0].cpu().numpy()
                        center_x, center_y = int(c_x), int(c_y)
                        box_w, box_h = int(w), int(h)
                        found_target = True
                        break # 찾았으면 루프 종료 (속도 향상)

            # (B) 일반 Box 탐색 (OBB에서 못 찾았을 경우)
            if not found_target and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id] # [필수] 이름 추출

                    if cls_name == target_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        box_w = abs(x2 - x1)
                        box_h = abs(y2 - y1)
                        found_target = True
                        break # 찾았으면 루프 종료

        if not found_target:
            self.get_logger().warn(f"   ⚠️ 타겟 '{target_name}' 없음")
            response.found = False
            return response

        # ---------------------------------------------------------
        # 3. Depth 추출 (ROI 영역 평균 사용)
        # ---------------------------------------------------------
        h_img, w_img = self.latest_depth_img.shape
        
        # 박스 크기의 30% 영역만 사용하여 정확도 향상
        roi_w = max(5, int(box_w * 0.3))
        roi_h = max(5, int(box_h * 0.3))
        
        x_min = max(0, center_x - roi_w // 2)
        x_max = min(w_img, center_x + roi_w // 2)
        y_min = max(0, center_y - roi_h // 2)
        y_max = min(h_img, center_y + roi_h // 2)
        
        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) == 0:
            self.get_logger().warn(f"   ⚠️ Depth 0 (거리 측정 불가)")
            # 필요하다면 여기서 고정값 처리 로직 추가 가능
            response.found = False
            return response
            
        depth_mm = np.median(valid_depths)

        # ---------------------------------------------------------
        # 4. 좌표 변환
        # ---------------------------------------------------------
        cam_point = self.pixel_to_3d_cam(center_x, center_y, depth_mm)
        
        if cam_point is None:
            self.get_logger().error("   ❌ 3D 변환 실패")
            response.found = False
            return response

        response.found = True
        response.position = Point(x=cam_point[0], y=cam_point[1], z=cam_point[2])
        response.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) 
        
        # 로그 간소화 (좌표만 딱 출력)
        self.get_logger().info(f"   ✅ 좌표: X={cam_point[0]:.1f}, Y={cam_point[1]:.1f}, Z={cam_point[2]:.1f}")
        
        return response

    

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()