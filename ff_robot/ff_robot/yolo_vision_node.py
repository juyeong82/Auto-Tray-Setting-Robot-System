#!/usr/bin/env python3
# yolo_vision_node.py (Final Optimized with Debug Publisher)

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
from geometry_msgs.msg import Point, Quaternion

# 거리 측정 실패 시 사용할 기본값 (60cm)
FIXED_DEPTH_MM = 600.0 

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        self.get_logger().info("=========================================")
        self.get_logger().info("🚀 YOLO Vision Node (Debug Pub Mode)")
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

        # 5. [추가됨] 디버깅용 이미지 발행 퍼블리셔
        # 큐 사이즈 10, 토픽 이름: /yolo_debug_image
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
            self.get_logger().info("✅ 카메라 파라미터 수신 완료")

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

        # 추론 실행
        results = self.model(self.latest_color_img, verbose=False, conf=0.25)
        
        found_target = False
        cx, cy, w, h = 0, 0, 0, 0
        rotation_deg = 0.0
        confidence = 0.0
        detected_names = []

        # 결과 파싱
        for r in results:
            if r.obb is not None:
                for box in r.obb:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    detected_names.append(cls_name)
                    if cls_name == target_name:
                        c_x, c_y, bw, bh, rot = box.xywhr[0].cpu().numpy()
                        cx, cy, w, h = int(c_x), int(c_y), int(bw), int(bh)
                        rotation_deg = self.calculate_obb_rotation(box)
                        confidence = float(box.conf[0])
                        found_target = True
                        break 
            if not found_target and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    detected_names.append(cls_name)
                    if cls_name == target_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        w, h = abs(x2-x1), abs(y2-y1)
                        confidence = float(box.conf[0])
                        found_target = True
                        break

        # ==========================================================
        # [추가된 부분] 디버깅 이미지 발행 로직 (조건부 실행)
        # ==========================================================
        if self.debug_image_pub.get_subscription_count() > 0:
            try:
                # YOLO가 박스/라벨을 그린 이미지를 생성 (.plot() 메서드)
                annotated_frame = results[0].plot()
                
                # ROS Image 메시지로 변환 후 발행
                debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                self.debug_image_pub.publish(debug_msg)
                # self.get_logger().info("📸 디버깅 이미지 발행됨") # 너무 시끄러우면 주석 처리
            except Exception as e:
                self.get_logger().warn(f"디버깅 이미지 발행 중 오류: {e}")
        # ==========================================================

        self.get_logger().info(f"   👀 감지된 물체: {list(set(detected_names))}")

        if not found_target:
            self.get_logger().warn(f"   ⚠️ 타겟 '{target_name}' 없음")
            response.found = False
            return response

        # Depth 추출 및 변환
        h_img, w_img = self.latest_depth_img.shape
        roi_w, roi_h = max(5, int(w * 0.3)), max(5, int(h * 0.3))
        x_min = max(0, cx - roi_w // 2)
        x_max = min(w_img, cx + roi_w // 2)
        y_min = max(0, cy - roi_h // 2)
        y_max = min(h_img, cy + roi_h // 2)
        
        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            depth_mm = float(np.median(valid_depths))
            self.get_logger().info(f"   📏 측정 거리: {depth_mm:.1f} mm")
        else:
            depth_mm = FIXED_DEPTH_MM
            self.get_logger().warn(f"   ⚠️ Depth 0 -> 고정값 {FIXED_DEPTH_MM}mm 사용")

        cam_point = self.pixel_to_3d_cam(cx, cy, depth_mm)
        if cam_point is None:
            self.get_logger().error("   ❌ 3D 변환 실패")
            response.found = False
            return response

        # ==========================================================
        # [수정됨] 디버깅 이미지 발행 로직 (텍스트 + 중심점 추가)
        # ==========================================================
        if self.debug_image_pub.get_subscription_count() > 0:
            try:
                # 1. YOLO가 기본 박스/라벨을 그린 이미지 가져오기
                annotated_frame = results[0].plot()

                # 2. 중심점 표시 (빨간색 점)
                # (이미지, 중심좌표, 반지름, 색상BGR, 두께(-1은 채움))
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                # 3. Z값(Depth) 텍스트 표시
                text = f"Z: {depth_mm:.1f} mm"
                text_pos = (cx - 40, cy + 30)  # 중심보다 약간 아래, 왼쪽으로 살짝 이동

                # 글씨가 잘 보이게 검은색 테두리(두께 4)를 먼저 그림
                cv2.putText(annotated_frame, text, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                
                # 그 위에 노란색 글씨(두께 2)를 덮어씀
                cv2.putText(annotated_frame, text, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 4. ROS 메시지 변환 및 발행
                debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                self.debug_image_pub.publish(debug_msg)
                
            except Exception as e:
                self.get_logger().warn(f"디버깅 이미지 발행 중 오류: {e}")
        # ==========================================================

        response.found = True
        response.position = Point(x=cam_point[0], y=cam_point[1], z=cam_point[2])
        response.rx = 0.0
        response.ry = 180.0
        response.rz = rotation_deg
        response.width = float(w)
        response.height = float(h)
        response.confidence = confidence
        
        self.get_logger().info(f"   ✅ 좌표 반환: X={cam_point[0]:.3f}, Y={cam_point[1]:.3f}, Z={cam_point[2]:.3f}")
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
        # cv2.destroyAllWindows() # 이제 창을 띄우지 않으므로 사실상 불필요하지만 안전을 위해 남둠

if __name__ == '__main__':
    main()