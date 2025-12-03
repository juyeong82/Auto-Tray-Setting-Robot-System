#!/usr/bin/env python3
# yolo_vision_node.py
# [Updated] Eye-to-Hand Base 좌표 변환 로그 출력 기능 추가

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
        try:
            self.declare_parameter('use_sim_time', True)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass
        self.get_logger().info("🚀 YOLO Vision Node (Eye-to-Hand Debug Log Added)")
        self.get_logger().info("=========================================")
        
        # 1. 모델 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "models", "best.pt")
        
        # [추가] Eye-to-Hand 캘리브레이션 행렬 로드 (로그 확인용)
        self.T_cam2base = None
        calib_path = os.path.join(current_dir, "T_cam2base.npy")
        if os.path.exists(calib_path):
            try:
                self.T_cam2base = np.load(calib_path)
                self.get_logger().info(f"✅ [Debug] 캘리브레이션 파일 로드 완료: {calib_path}")
            except Exception as e:
                self.get_logger().error(f"❌ [Debug] 캘리브레이션 파일 로드 에러: {e}")
        else:
            self.get_logger().warn(f"⚠️ [Debug] T_cam2base.npy 없음. Base 좌표 로그 출력 불가.")

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

        # 4. 구독 설정
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
        while self.latest_color_img is None or self.latest_depth_img is None:
            if time.time() - wait_start > 5.0: 
                self.get_logger().error("❌ [Timeout] 이미지 수신 실패")
                response.found = False
                return response
            time.sleep(0.01)

        results = self.model(self.latest_color_img, verbose=False, conf=0.25)
        candidates = []
        
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
                        
                        roi_w, roi_h = max(5, int(bw * 0.3)), max(5, int(bh * 0.3))
                        x_min = max(0, cx - roi_w // 2); x_max = min(w_img, cx + roi_w // 2)
                        y_min = max(0, cy - roi_h // 2); y_max = min(h_img, cy + roi_h // 2)
                        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
                        valid_depths = roi[roi > 0]
                        depth_mm = float(np.median(valid_depths)) if len(valid_depths) > 0 else FIXED_DEPTH_MM

                        cam_point = self.pixel_to_3d_cam(cx, cy, depth_mm)
                        if cam_point is not None:
                            candidates.append({
                                'x': cam_point[0], 'y': cam_point[1], 'z': cam_point[2],
                                'rz': rotation_deg, 'w': float(bw), 'h': float(bh), 'conf': conf,
                                'px': cx, 'py': cy
                            })

            # 2. 일반 Box 검색
            if not candidates and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name == target_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        w, h = abs(x2-x1), abs(y2-y1)
                        conf = float(box.conf[0])
                        
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

        if candidates:
            # X^2 + Y^2 거리 기준 정렬
            candidates.sort(key=lambda c: (c['x']**2 + c['y']**2))
            best = candidates[0]
            
            response.found = True
            response.position = Point(x=best['x'], y=best['y'], z=best['z'])
            response.rx = 0.0
            response.ry = 180.0
            response.rz = best['rz']
            response.width = best['w']
            response.height = best['h']
            response.confidence = best['conf']
            
            dist_sq = best['x']**2 + best['y']**2
            self.get_logger().info(f"   ✅ 선택됨(Cam): X={best['x']:.2f}, Y={best['y']:.2f}, Z={best['z']:.2f}")

            # [추가] 로봇 Base 좌표계 기준 변환 좌표 출력 (디버깅용)
            if self.T_cam2base is not None:
                try:
                    # 미터(m) 단위 -> 밀리미터(mm) 변환 (캘리브레이션은 mm 단위 가정)
                    p_cam = np.array([best['x']*1000, best['y']*1000, best['z']*1000, 1.0])
                    p_base = self.T_cam2base @ p_cam
                    
                    # 최종 출력
                    bx, by, bz = p_base[0], p_base[1], p_base[2]
                    self.get_logger().info(f"   🤖 [Base Coord Preview] X={bx:.1f}, Y={by:.1f}, Z={bz:.1f}")
                except Exception as e:
                    self.get_logger().warn(f"   ⚠️ Base 좌표 변환 중 오류: {e}")

            # 디버깅 이미지
            if self.debug_image_pub.get_subscription_count() > 0:
                try:
                    annotated_frame = results[0].plot()
                    cv2.circle(annotated_frame, (best['px'], best['py']), 8, (0, 0, 255), -1)
                    
                    text = f"Dist: {math.sqrt(dist_sq):.2f}m"
                    cv2.putText(annotated_frame, text, (best['px']-40, best['py']-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                    self.debug_image_pub.publish(debug_msg)
                except: pass
        else:
            response.found = False
            self.get_logger().warn(f"   ⚠️ 타겟 '{target_name}' 없음")
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