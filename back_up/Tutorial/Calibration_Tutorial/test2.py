import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG

import time
import numpy as np
import DR_init
import sys

# =========================================================
# [설정] 안전 모드 (True: 이동 안 함, 좌표만 출력)
SAFE_MODE = True 
# =========================================================

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")
        
        cv2.namedWindow("Webcam")
        self.img_node = ImgNode()
        
        self.intrinsics = None
        # Eye-to-Hand 캘리브레이션 결과 행렬 로드
        try:
            # 파일 이름이 T_cam2base.npy인지 확인 필수
            # self.T_cam2base = np.load("T_cam2base.npy")
            self.T_cam2base = np.load("/home/juyeong/ros2_ws/src/back_up/Tutorial/Calibration_Tutorial/T_cam2base.npy")
            
            self.get_logger().info(f"Calibration Matrix Loaded:\n{self.T_cam2base}")
        except FileNotFoundError:
            self.get_logger().error("T_cam2base.npy 파일을 찾을 수 없습니다. 단위 행렬을 사용합니다.")
            self.T_cam2base = np.eye(4)

        # 카메라 내부 파라미터 대기
        self.get_logger().info("--- Waiting for Camera Intrinsics ---")
        while rclpy.ok() and self.intrinsics is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1) 
            intrinsics = self.img_node.get_camera_intrinsic()
            
            if intrinsics is not None and "fx" in intrinsics: 
                self.intrinsics = intrinsics
                self.get_logger().info(f"Camera Intrinsics Loaded: {self.intrinsics}")
                break
            
            self.get_logger().warn("Waiting for camera info topic...")
            time.sleep(0.5)

        # 마우스 콜백 설정
        cv2.setMouseCallback("Webcam", self.mouse_callback)
        self.get_logger().info("준비 완료. 화면을 클릭하여 좌표를 확인하세요.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_frame = self.img_node.get_depth_frame()
            
            # 깊이 이미지 유효성 체크
            if depth_frame is None:
                self.get_logger().warn("Depth frame is None")
                return

            print("\n" + "="*50)
            print(f"[클릭] 이미지 픽셀 좌표: (u={x}, v={y})")
            
            # Depth 값 추출
            z = self.get_depth_value(x, y, depth_frame)
            if z is None or z <= 0:
                self.get_logger().warn("깊이 값을 읽을 수 없습니다 (0 또는 유효하지 않음).")
                print("="*50)
                return
            
            print(f" -> 측정된 깊이(Z): {z} mm")

            # 1. 2D 픽셀 -> 3D 카메라 좌표계 변환
            camera_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            print(f" -> 카메라 기준 좌표 (Cam):  X={camera_pos[0]:.2f}, Y={camera_pos[1]:.2f}, Z={camera_pos[2]:.2f}")

            # 2. 3D 카메라 -> 3D 로봇 베이스 좌표계 변환 (Eye-to-Hand)
            robot_pos = self.transform_to_base(camera_pos)
            print(f" -> 로봇 베이스 좌표 (Base): X={robot_pos[0]:.2f}, Y={robot_pos[1]:.2f}, Z={robot_pos[2]:.2f}")
            print("="*50)

            # SAFE_MODE가 False일 때만 움직임 (현재는 무조건 True라 안 움직임)
            if not SAFE_MODE:
                self.move_robot(robot_pos)

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        # 핀홀 카메라 모델 역투영
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z
        return (camera_x, camera_y, camera_z)

    def transform_to_base(self, camera_coords):
        # [중요] Eye-to-Hand 변환 수식
        # P_base = T_cam2base * P_camera
        # 로봇의 현재 자세(get_current_posx)는 전혀 필요 없음 (카메라가 고정되어 있으므로)
        
        # 동차 좌표계로 변환 [x, y, z, 1]
        p_cam = np.append(np.array(camera_coords), 1)
        
        # 행렬 곱셈
        p_base = np.dot(self.T_cam2base, p_cam)
        
        return p_base[:3] # [x, y, z] 반환

    def get_depth_value(self, center_x, center_y, depth_frame):
        height, width = depth_frame.shape
        if 0 <= center_x < width and 0 <= center_y < height:
            return depth_frame[center_y, center_x]
        return None

    def update_image(self):
        rclpy.spin_once(self.img_node, timeout_sec=0.001)
        img = self.img_node.get_color_frame()
        if img is not None:
            # 화면에 현재 모드 표시
            cv2.putText(img, "VIEW ONLY MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Webcam", img)
        return img
    
    def move_robot(self, target_pos):
        # 이동 로직이 필요하다면 여기에 작성 (현재는 안전 모드로 호출되지 않음)
        pass

if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    # DSR 관련 임포트 (초기화 구문 유지)
    try:
        from DSR_ROBOT2 import *
        from DR_common2 import *
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")

    test_node = TestNode()

    while rclpy.ok():
        test_node.update_image()
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cv2.destroyAllWindows()
    test_node.destroy_node()
    rclpy.shutdown()