import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG
from ultralytics import YOLO

import time
import numpy as np
import DR_init

# --- 로봇 및 그리퍼 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


class AutomationNode(Node):
    def __init__(self):
        super().__init__("automation_node")
        
        cv2.namedWindow("Webcam")
        self.img_node = ImgNode()
        
        # 1. YOLO 모델 로드 (사용자 모델 best.pt 사용)
        try:
            self.model = YOLO("best.pt") 
            self.get_logger().info("✅ YOLO Model Loaded Successfully!")
            # 모델이 아는 클래스 이름 출력 (입력 시 참고용)
            self.get_logger().info(f"Available Classes: {self.model.names}")
        except Exception as e:
            self.get_logger().error(f"YOLO Model Load Failed: {e}")

        self.intrinsics = None
        self.JReady = posj([-3.98, 7.54, 64.19, -6.33, 106.34, -3.48])
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        self.get_logger().info("--- Waiting for Camera Intrinsics ---")
        while rclpy.ok() and self.intrinsics is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1) 
            intrinsics = self.img_node.get_camera_intrinsic()
            if intrinsics is not None and "fx" in intrinsics: 
                self.intrinsics = intrinsics
                break
            time.sleep(0.5)

        try:
            self.gripper2cam = np.load("T_gripper2camera.npy")
        except FileNotFoundError:
            self.get_logger().error("T_gripper2camera.npy not found. Using Identity.")
            self.gripper2cam = np.eye(4)
        

    def detect_and_pick(self, target_name):
        """
        [수정됨] 매개변수로 받은 'target_name'과 일치하는 객체만 찾습니다.
        """
        color_frame = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()
        
        # 입력받은 타겟 이름 (혹시 모르니 소문자로 변환)
        TARGET_CLASS = target_name.lower().strip()

        if color_frame is None or depth_frame is None:
            self.get_logger().warn("No frame received")
            return

        self.get_logger().info(f"🔍 Searching for target: '{TARGET_CLASS}'...")

        # 1. 예측 실행
        results = self.model.predict(color_frame, conf=0.5, verbose=False)
        
        if len(results[0].boxes) == 0:
            self.get_logger().warn("No objects detected at all.")
            return

        # ==========================================================
        # [핵심] 입력받은 이름과 같은 객체 찾기
        # ==========================================================
        target_box = None
        found_target = False

        print("-" * 30)
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = results[0].names[class_id]
            conf = box.conf[0].item()
            
            print(f"👀 Scan: '{class_name}' ({conf:.2f})")

            # 입력한 이름과 감지된 이름이 같으면 선택
            if class_name == TARGET_CLASS:
                target_box = box
                found_target = True
                print(f"   ✅ Target Match! Found '{TARGET_CLASS}' ({conf:.2f})")
                break 
            else:
                pass # 다른 물건은 무시

        print("-" * 30)

        if not found_target or target_box is None:
            self.get_logger().warn(f"🚫 화면에 '{TARGET_CLASS}'가 없습니다! 이름을 다시 확인하세요.")
            return

        # 좌표 계산
        x_center, y_center, w, h = target_box.xywh[0].cpu().numpy()
        x, y = int(x_center), int(y_center)
        
        print(f"🎯 Locked Target: {TARGET_CLASS} at Pixel ({x}, {y})")

        # 뎁스값 추출
        z = self.get_depth_value(x, y, depth_frame)
        
        if z is None or z == 0:
            self.get_logger().warn("Depth invalid, skipping...")
            return

        # 좌표 변환
        camera_pos = self.get_camera_pos(x, y, z, self.intrinsics)
        robot_pos = self.transform_to_base(camera_pos)
        
        print(f"   📷 Camera Coord: {camera_pos}")
        print(f"   🤖 Robot Base Coord: {robot_pos}")

        # 시각화 (화면에 타겟 이름 표시)
        cv2.circle(color_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(color_frame, f"TARGET: {TARGET_CLASS}", (x, y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Webcam", color_frame)
        cv2.waitKey(100)

        # 로봇 실행
        self.execute_pick_sequence(robot_pos)

    def execute_pick_sequence(self, robot_coordinate):
        x, y, z = robot_coordinate

        # JReady 이동 (안전)
        movej(self.JReady, vel=VELOCITY, acc=ACC)
        
        # 픽앤드롭 실행
        self.pick_and_drop(x, y, z)
        
        # 복귀
        movej(self.JReady, vel=VELOCITY, acc=ACC)
        print("✅ Task Finished. Ready for next input.")

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z
        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        coord = np.append(np.array(camera_coords), 1)
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        
        # [옵션] 좌표 미세 보정 (필요시 주석 해제하여 사용) -offset
        final_x, final_y, final_z = td_coord[:3]
        final_x += 0.0
        final_y += 20.0
        return [final_x, final_y, final_z]

    def get_depth_value(self, center_x, center_y, depth_frame):
        height, width = depth_frame.shape
        roi_size = 5
        half_size = roi_size // 2

        if (center_x - half_size < 0 or center_x + half_size >= width or 
            center_y - half_size < 0 or center_y + half_size >= height):
            return None

        roi = depth_frame[center_y - half_size : center_y + half_size + 1,
                          center_x - half_size : center_x + half_size + 1]
        
        valid_depths = roi[roi > 0]
        if len(valid_depths) == 0:
            return None
        
        return np.median(valid_depths)

    def pick_and_drop(self, x, y, z):
        current_pos = get_current_posx()[0]
        self.gripper.open_gripper()
        wait(1)

        # 위치 설정 (Z+200: 잡는 높이, Z+300: 접근 높이)
        target_grip_pos = posx([x, y, z + 170, current_pos[3], current_pos[4], current_pos[5]])
        target_approach_pos = posx([x, y, z + 300, current_pos[3], current_pos[4], current_pos[5]])
        
        self.get_logger().info(f"📍 Target Grip: {target_grip_pos}")

        movel(target_approach_pos, vel=VELOCITY, acc=ACC) # 접근
        wait(0.5)
        movel(target_grip_pos, vel=VELOCITY, acc=ACC) # 하강
        wait(0.5)
        
        self.gripper.close_gripper(150) # 잡기
        wait(1)

        movel([0, 0, 150, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) # 상승
        wait(1)
        
        movel(target_grip_pos, vel=VELOCITY, acc=ACC) # (테스트용) 놓기
        self.gripper.open_gripper()
        wait(1)
        
        movel(target_approach_pos, vel=VELOCITY, acc=ACC) # 복귀

    def update_view(self):
        rclpy.spin_once(self.img_node, timeout_sec=0.001)
        img = self.img_node.get_color_frame()
        if img is not None:
            cv2.imshow("Webcam", img) 
        return img


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_yolo_automation", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait, DR_MV_MOD_REL
        from DR_common2 import posx, posj
    except ImportError:
        exit(True)

    auto_node = AutomationNode()

    # 1. 시작 시 JReady 이동
    print("🚀 Initializing Robot Position to JReady...")
    movej(auto_node.JReady, vel=VELOCITY, acc=ACC)
    print("✅ Initialization Complete!")

    print("\n" + "="*60)
    print("   [Enter]: 입력 모드 시작 (터미널에 찾을 물건 입력)")
    print("   [ESC]: 종료")
    print("="*60 + "\n")

    while rclpy.ok():
        auto_node.update_view()
        
        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        
        # 엔터키(Enter) = 13
        if key == 13: 
            # 2. 터미널에서 사용자 입력 받기
            print("\n" + "-"*40)
            target_input = input(">> 찾을 물건 이름을 입력하세요 (예: apple, lemon): ")
            print("-"*40)
            
            if target_input.strip(): # 빈 칸이 아니면 실행
                auto_node.detect_and_pick(target_input)
            else:
                print("⚠️ 입력이 없습니다. 다시 시도하세요.")
        
        elif key == 27: # ESC
            break

    cv2.destroyAllWindows()
    auto_node.destroy_node()
    rclpy.shutdown()