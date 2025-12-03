import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from onrobot import RG
import time
import numpy as np
import DR_init

# ==========================================
# [설정] 동작 및 오프셋 파라미터
VELOCITY, ACC = 40, 40   # 이동 속도
<<<<<<< HEAD
HOVER_HEIGHT = 0.0     # Z축: 바닥에서 얼마나 띄울지 (mm)
WAIT_TIME = 10.0          # 대기 시간 (초)
=======
HOVER_HEIGHT = 10     # Z축: 바닥에서 얼마나 띄울지 (mm)
WAIT_TIME = 0          # 대기 시간 (초)
>>>>>>> calibration

# [중요] 좌표 오프셋 설정 (mm 단위)
OFFSET_X = 0   # X축 -50mm 이동
OFFSET_Y = 0   # Y축 +100mm 이동
# ==========================================

# Robot Config
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

class TestOffsetMoveNode(Node):
    def __init__(self):
        super().__init__("test_offset_move")
        
        cv2.namedWindow("Webcam")
        self.img_node = ImgNode()
        self.intrinsics = None
        
        # 홈 포즈 (대기 위치)
        self.JReady = posj([0, 0, 90, 0, 90, 0])

        try:
<<<<<<< HEAD
            self.T_cam2base = np.load("T_cam2base.npy")
=======
            # self.T_cam2base = np.load("T_cam2base.npy")
            self.T_cam2base = np.load("/home/juyeong/ros2_ws/src/back_up/Tutorial/Calibration_Tutorial/T_cam2base.npy")
>>>>>>> calibration
            self.get_logger().info("✅ T_cam2base.npy 로드 성공")
        except FileNotFoundError:
            self.get_logger().error("🚨 T_cam2base.npy 파일 없음")
            exit()

        self.get_logger().info("--- 카메라 파라미터 수신 대기 ---")
        while rclpy.ok() and self.intrinsics is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1) 
            intrinsics = self.img_node.get_camera_intrinsic()
            if intrinsics is not None and "fx" in intrinsics: 
                self.intrinsics = intrinsics
                self.get_logger().info("✅ 카메라 파라미터 수신 완료")
                break
            time.sleep(0.5)

        cv2.setMouseCallback("Webcam", self.mouse_callback)
        self.get_logger().info("🚀 준비 완료. 화면을 클릭하세요.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_frame = self.img_node.get_depth_frame()
            if depth_frame is None: return

            z = self.get_depth_value(x, y, depth_frame)
            if z is None or z <= 0:
                self.get_logger().warn("⚠️ 유효하지 않은 깊이 값")
                return

            # --- 좌표 계산 ---
            camera_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            robot_pos = self.transform_to_base(camera_pos)

            print("\n" + "="*60)
            print(f"📍 [클릭 원본 좌표]")
            print(f" - 이미지 (Pixel) : u={x}, v={y}")
            print(f" - 로봇 베이스     : X={robot_pos[0]:.1f}, Y={robot_pos[1]:.1f}, Z={robot_pos[2]:.1f}")
            
            # --- 오프셋 이동 실행 ---
            self.move_offset_and_return(robot_pos)

    def move_offset_and_return(self, origin_xyz):
        ox, oy, oz = origin_xyz
        
        # 현재 로봇 회전 상태 유지
        curr_pos = get_current_posx()[0] 
        rx, ry, rz = curr_pos[3], curr_pos[4], curr_pos[5]

        # [핵심] 오프셋 적용 (원본 클릭 위치 + 설정값)
        target_x = ox + OFFSET_X
        target_y = oy + OFFSET_Y
        target_z = oz + HOVER_HEIGHT # 높이는 여전히 안전하게 띄움

        print(f"🎯 [최종 목표 좌표 (오프셋 적용)]")
        print(f" - X: {ox:.1f} -> {target_x:.1f} (Diff: {OFFSET_X})")
        print(f" - Y: {oy:.1f} -> {target_y:.1f} (Diff: {OFFSET_Y})")
        print(f" - Z: {oz:.1f} -> {target_z:.1f} (Height: +{HOVER_HEIGHT})")
        print("="*60)

        # 좌표 생성
        pos_target = posx([target_x, target_y, target_z, rx, ry, rz])
        
        print(f"🚀 이동 시작...")
        
        # 1. 오프셋 위치로 이동
        movel(pos_target, vel=VELOCITY, acc=ACC)
        
        print(f"⏳ {WAIT_TIME}초 대기...")
        wait(WAIT_TIME)

        # 2. 홈 복귀
        print("🏠 홈으로 복귀")
<<<<<<< HEAD
        movej(self.JReady, vel=VELOCITY, acc=ACC)
=======
        # movej(self.JReady, vel=VELOCITY, acc=ACC)
>>>>>>> calibration
        print("✅ 완료.\n")

    def get_camera_pos(self, u, v, z, intrin):
        fx, fy = intrin["fx"], intrin["fy"]
        cx, cy = intrin["ppx"], intrin["ppy"]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return [x, y, z]

    def transform_to_base(self, cam_xyz):
        p_cam = np.append(np.array(cam_xyz), 1)
        p_base = np.dot(self.T_cam2base, p_cam)
        return p_base[:3]

    def get_depth_value(self, u, v, depth_frame):
        h, w = depth_frame.shape
        if 0 <= u < w and 0 <= v < h:
            return depth_frame[v, u]
        return None
    
    def update_image(self):
        rclpy.spin_once(self.img_node, timeout_sec=0.001)
        img = self.img_node.get_color_frame()
        if img is not None:
            info_text = f"Offset: X{OFFSET_X} Y+{OFFSET_Y}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Webcam", img)
        return img

if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_offset_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import *
        from DR_common2 import *
    except ImportError as e:
        print(f"DSR Import Error: {e}")

    test_node = TestOffsetMoveNode()

    while rclpy.ok():
        test_node.update_image()
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    test_node.destroy_node()
    rclpy.shutdown()