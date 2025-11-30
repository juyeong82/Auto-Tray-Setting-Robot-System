#!/usr/bin/env python3
# yolo_vision_node.py (Tray Detection + Euler Angle Ver)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import math

from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point

# =========================================================
# [설정] 깊이 센서 오류 시 고정값
# =========================================================
FIXED_DEPTH_MM = 600.0 

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        
        self.declare_parameter('use_sim_time', True)  # Gazebo 시간 사용
        
        self.get_logger().info("=========================================")
        self.get_logger().info("🚀 YOLO Vision Node (Tray + Item Detection)")
        self.get_logger().info("=========================================")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "models", "best.pt")
        
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.camera_intrinsics = None 
        
        try:
            self.model = YOLO(self.model_path) 
            self.get_logger().info("✅ YOLO 모델 로드 완료")
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")

        qos_profile = qos_profile_sensor_data
        self.cb_group = ReentrantCallbackGroup()

        # [수정] 토픽 이름을 Gazebo에 맞게 변경
        self.create_subscription(Image, '/dsr01/camera/color/image_raw', 
                                self.color_callback, qos_profile, callback_group=self.cb_group)
        self.create_subscription(Image, '/dsr01/camera/depth/image_rect_raw', 
                                self.depth_callback, qos_profile, callback_group=self.cb_group)
        self.create_subscription(CameraInfo, '/dsr01/d435i/camera_info', 
                                self.info_callback, qos_profile, callback_group=self.cb_group)

        self.srv = self.create_service(DetectObject, '/dsr01/detect_object', 
                                       self.handle_detect_object, callback_group=self.cb_group)
        self.get_logger().info("👀 서비스 서버 준비 완료")

    def color_callback(self, msg):
        try: 
            self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: 
            pass

    def depth_callback(self, msg):
        try: 
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except: 
            pass

    def info_callback(self, msg):
        if self.camera_intrinsics is None:
            K = msg.k
            self.camera_intrinsics = {
                'fx': K[0], 
                'fy': K[4], 
                'ppx': K[2], 
                'ppy': K[5]
            }

    def pixel_to_3d_cam(self, u, v, depth_mm):
        """픽셀 좌표 → 3D 카메라 좌표 변환"""
        if self.camera_intrinsics is None: 
            return None
        z = depth_mm / 1000.0 
        if z <= 0: 
            return None
        x = (u - self.camera_intrinsics['ppx']) * z / self.camera_intrinsics['fx']
        y = (v - self.camera_intrinsics['ppy']) * z / self.camera_intrinsics['fy']
        return np.array([x, y, z])

    def calculate_obb_rotation(self, box):
        """
        OBB의 회전각을 계산 (Yaw만, Euler rz)
        YOLO OBB는 xywhr 형식: [center_x, center_y, width, height, rotation_radians]
        """
        try:
            # OBB에서 회전값 추출 (라디안)
            rotation_rad = float(box.xywhr[0][4].cpu().numpy())
            
            # 라디안 → 도(degree) 변환
            rotation_deg = math.degrees(rotation_rad)
            
            # -180 ~ 180 범위로 정규화
            while rotation_deg > 180:
                rotation_deg -= 360
            while rotation_deg < -180:
                rotation_deg += 360
                
            return rotation_deg
            
        except Exception as e:
            self.get_logger().warn(f"회전 계산 실패: {e}")
            return 0.0

    def handle_detect_object(self, request, response):
        target_name = request.target_object_id
        self.get_logger().info(f"🔎 YOLO 요청 수신: '{target_name}'")

        # 1. 이미지 대기
        wait_start = time.time()
        while self.latest_color_img is None or self.latest_depth_img is None:
            if time.time() - wait_start > 5.0: 
                self.get_logger().error(f"❌ [Timeout] 이미지 수신 실패")
                response.found = False
                return response
            time.sleep(0.01)
        
        self.get_logger().info("✅ 이미지 확보. 추론 시작.")

        # 2. YOLO 추론
        results = self.model(self.latest_color_img, verbose=False)
        
        debug_img = self.latest_color_img.copy()
        
        found_target = False
        center_x, center_y = 0, 0
        box_w, box_h = 20, 20
        rotation_deg = 0.0
        confidence = 0.0
        detected_names = []

        # 3. OBB 우선 검색
        for r in results:
            if r.obb is not None:
                for box in r.obb:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    detected_names.append(cls_name)
                    
                    # 모든 박스 그리기 (파란색)
                    pts = box.xyxyxyxy[0].cpu().numpy().reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(debug_img, [pts], True, (255, 0, 0), 2)
                    cv2.putText(debug_img, f"{cls_name} {conf:.2f}", 
                               (pts[0][0][0], pts[0][0][1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if cls_name == target_name:
                        c_x, c_y, w, h, rot = box.xywhr[0].cpu().numpy()
                        center_x, center_y = int(c_x), int(c_y)
                        box_w, box_h = int(w), int(h)
                        rotation_deg = self.calculate_obb_rotation(box)
                        confidence = conf
                        found_target = True
                        
                        # 타겟 중심점 찍기 (빨간색)
                        cv2.circle(debug_img, (center_x, center_y), 8, (0, 0, 255), -1)
                        cv2.putText(debug_img, f"TARGET rot={rotation_deg:.1f}deg", 
                                   (center_x+10, center_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 4. 일반 박스 검색 (OBB 없을 때)
            elif r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    detected_names.append(cls_name)
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    if cls_name == target_name:
                        center_x = int((x1+x2)/2)
                        center_y = int((y1+y2)/2)
                        box_w, box_h = abs(x2-x1), abs(y2-y1)
                        confidence = conf
                        found_target = True
                        
                        cv2.circle(debug_img, (center_x, center_y), 8, (0, 0, 255), -1)

        self.get_logger().info(f"   👀 화면 감지: {list(set(detected_names))}")

        if not found_target:
            self.get_logger().warn(f"   ⚠️ 타겟 '{target_name}' 없음")
            cv2.imshow(f"Check: {target_name}", debug_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            
            response.found = False
            return response

        # 5. Depth 추출
        h_img, w_img = self.latest_depth_img.shape
        roi_w = max(5, int(box_w * 0.3))
        roi_h = max(5, int(box_h * 0.3))
        x_min = max(0, center_x - roi_w // 2)
        x_max = min(w_img, center_x + roi_w // 2)
        y_min = max(0, center_y - roi_h // 2)
        y_max = min(h_img, center_y + roi_h // 2)
        
        roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) == 0:
            self.get_logger().warn(f"   ⚠️ Depth 0 -> 고정값 {FIXED_DEPTH_MM}mm 사용")
            depth_mm = FIXED_DEPTH_MM
        else:
            depth_mm = float(np.median(valid_depths))
            self.get_logger().info(f"   📏 측정 거리: {depth_mm:.1f} mm")

        # 6. 좌표 변환
        cam_point = self.pixel_to_3d_cam(center_x, center_y, depth_mm)
        if cam_point is None:
            self.get_logger().error("   ❌ 3D 변환 실패")
            response.found = False
            return response

        # 7. 사용자 확인 (Debug)
        info_txt = f"X:{cam_point[0]:.2f} Y:{cam_point[1]:.2f} Z:{cam_point[2]:.2f} Rz:{rotation_deg:.1f}deg"
        cv2.putText(debug_img, info_txt, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, "Press [ENTER] to Go, [ESC] to Cancel", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow(f"Confirm: {target_name}", debug_img)
        self.get_logger().warn("⌨️  [Wait] 이미지 창에서 Enter를 누르세요! (ESC: 취소)")
        
        while True:
            key = cv2.waitKey(100)
            if key == 13:  # Enter
                self.get_logger().info("✅ 사용자 확인 완료. 좌표 전송.")
                break
            elif key == 27:  # ESC
                self.get_logger().warn("🚫 사용자 취소.")
                cv2.destroyAllWindows()
                response.found = False
                return response
                
        cv2.destroyAllWindows()

        # 8. 응답 구성
        response.found = True
        response.position = Point(x=cam_point[0], y=cam_point[1], z=cam_point[2])
        
        # [핵심] Euler 각도로 반환
        response.rx = 0.0  # Roll (현재는 0)
        response.ry = 180.0  # Pitch (그리퍼 아래 방향)
        response.rz = rotation_deg  # Yaw (OBB 회전값!)
        
        response.width = float(box_w)
        response.height = float(box_h)
        response.confidence = confidence
        
        self.get_logger().info(f"   ✅ 좌표 반환: Pos={cam_point}, Rot(rz)={rotation_deg:.1f}deg")
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
