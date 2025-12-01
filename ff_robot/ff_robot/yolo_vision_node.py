#!/usr/bin/env python3
# yolo_vision_node.py
# [Final Fix] 사용자의 작동하는 코드를 기반으로 '가까운 순 정렬' 로직만 추가

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import math

from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point

# 거리 측정 실패 시 사용할 기본값 (60cm)
FIXED_DEPTH_MM = 600.0 

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        self.get_logger().info("=========================================")
        # [안전장치] 파라미터 중복 선언 방지
        try:
            self.declare_parameter('use_sim_time', True)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass
        self.get_logger().info("🚀 YOLO Vision Node (Sorted by Nearest X)")
        self.get_logger().info("=========================================")
        
        # 1. 모델 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "models", "best.pt")
        
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.camera_intrinsics = None 
        
        try:
            self.model = YOLO(self.model_path) 
            self.get_logger().info(f"✅ YOLO 모델 로드 완료: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")

        # 2. QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_depth = qos_profile_sensor_data
        
        # 3. 콜백 그룹
        self.cb_group = ReentrantCallbackGroup()

        # 4. 구독 설정 (사용자 코드의 토픽명 유지)
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', 
            self.color_callback, qos_profile, callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', 
            self.depth_callback, qos_depth, callback_group=self.cb_group
        )
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', 
            self.info_callback, qos_profile, callback_group=self.cb_group
        )

        # 5. 디버깅용 이미지 발행 퍼블리셔
        self.debug_image_pub = self.create_publisher(
            Image, 'yolo_debug_image', 10, callback_group=self.cb_group
        )

        # 6. 서비스 서버 설정
        self.srv = self.create_service(
            DetectObject, '/dsr01/detect_object', 
            self.handle_detect_object, callback_group=self.cb_group
        )
        self.get_logger().info("👀 Vision Service Ready")

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

    def calculate_obb_rotation(self, box):
        try:
            rotation_rad = float(box.xywhr[0][4].cpu().numpy())
            rotation_deg = math.degrees(rotation_rad)
            while rotation_deg > 180: rotation_deg -= 360
            while rotation_deg < -180: rotation_deg += 360
            return rotation_deg
        except: return 0.0

    def handle_detect_object(self, request, response):
        target_name = request.target_object_id
        self.get_logger().info(f"🔎 요청: '{target_name}'")

        wait_start = time.time()
        # 타임아웃 5초
        while self.latest_color_img is None or self.latest_depth_img is None:
            if time.time() - wait_start > 5.0: 
                self.get_logger().error("❌ [Timeout] 이미지 수신 실패 (토픽 확인 필요)")
                response.found = False
                return response
            time.sleep(0.01)

        # 추론 실행
        results = self.model(self.latest_color_img, verbose=False, conf=0.25)
        
        # [수정] 모든 후보를 담을 리스트
        candidates = []
        
        # 이미지 크기
        h_img, w_img = self.latest_depth_img.shape

        for r in results:
            # 1. OBB 검색
            if r.obb is not None:
                for box in r.obb:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    if cls_name == target_name:
                        c_x, c_y, bw, bh, rot = box.xywhr[0].cpu().numpy()
                        cx, cy = int(c_x), int(c_y)
                        rotation_deg = self.calculate_obb_rotation(box)
                        conf = float(box.conf[0])
                        
                        # Depth 처리
                        roi_w, roi_h = max(5, int(bw * 0.3)), max(5, int(bh * 0.3))
                        x_min = max(0, cx - roi_w // 2); x_max = min(w_img, cx + roi_w // 2)
                        y_min = max(0, cy - roi_h // 2); y_max = min(h_img, cy + roi_h // 2)
                        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
                        valid_depths = roi[roi > 0]
                        depth_mm = float(np.median(valid_depths)) if len(valid_depths) > 0 else FIXED_DEPTH_MM

                        # 3D 변환
                        cam_point = self.pixel_to_3d_cam(cx, cy, depth_mm)
                        if cam_point is not None:
                            candidates.append({
                                'x': cam_point[0], 'y': cam_point[1], 'z': cam_point[2],
                                'rz': rotation_deg, 'w': float(bw), 'h': float(bh), 'conf': conf,
                                'px': cx, 'py': cy
                            })

            # 2. 일반 Box 검색 (OBB 없을 때 대비)
            if not candidates and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name == target_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        w, h = abs(x2-x1), abs(y2-y1)
                        conf = float(box.conf[0])
                        
                        # Depth 처리
                        roi_w, roi_h = max(5, int(w * 0.3)), max(5, int(h * 0.3))
                        x_min = max(0, cx - roi_w // 2); x_max = min(w_img, cx + roi_w // 2)
                        y_min = max(0, cy - roi_h // 2); y_max = min(h_img, cy + roi_h // 2)
                        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
                        valid_depths = roi[roi > 0]
                        depth_mm = float(np.median(valid_depths)) if len(valid_depths) > 0 else FIXED_DEPTH_MM

                        cam_point = self.pixel_to_3d_cam(cx, cy, depth_mm)
                        if cam_point is not None:
                            candidates.append({
                                'x': cam_point[0], 'y': cam_point[1], 'z': cam_point[2],
                                'rz': 0.0, 'w': float(w), 'h': float(h), 'conf': conf,
                                'px': cx, 'py': cy
                            })

        # [결과 처리]
        if candidates:
            # X축(거리) 기준 오름차순 정렬 -> 가장 작은 값(가까운 것) 선택
            candidates.sort(key=lambda c: c['x'])
            best = candidates[0]
            
            response.found = True
            response.position = Point(x=best['x'], y=best['y'], z=best['z'])
            response.rx = 0.0
            response.ry = 180.0
            response.rz = best['rz']
            response.width = best['w']
            response.height = best['h']
            response.confidence = best['conf']
            
            self.get_logger().info(f"   ✅ 선택됨: X={best['x']:.3f}, Y={best['y']:.3f}, Z={best['z']:.3f}, Rz={best['rz']:.1f}")
            self.get_logger().info(f"   📊 (총 {len(candidates)}개 발견 중 최단거리 선택)")

            # 디버깅 이미지 발행
            if self.debug_image_pub.get_subscription_count() > 0:
                try:
                    annotated_frame = results[0].plot()
                    # 선택된 타겟 강조 (초록색 원)
                    cv2.circle(annotated_frame, (best['px'], best['py']), 8, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, "SELECTED", (best['px']-40, best['py']-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                    self.debug_image_pub.publish(debug_msg)
                except Exception as e:
                    self.get_logger().warn(f"디버깅 이미지 오류: {e}")
        else:
            response.found = False
            self.get_logger().warn(f"   ⚠️ 타겟 '{target_name}' 없음")
            # 실패 시에도 원본 이미지는 보내주면 디버깅에 도움됨
            if self.debug_image_pub.get_subscription_count() > 0:
                try:
                    annotated_frame = results[0].plot()
                    debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                    self.debug_image_pub.publish(debug_msg)
                except: pass

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

if __name__ == '__main__':
    main()