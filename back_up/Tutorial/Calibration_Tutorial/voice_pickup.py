import sys
import os


VOICE_PATH = '/home/rokey/ros2_ws/Tutorial/VoiceProcessing'
sys.path.append(VOICE_PATH)

from dotenv import load_dotenv
env_path = os.path.join(VOICE_PATH, '.env')
load_dotenv(dotenv_path=env_path)

import os
openai_api_key = os.getenv("OPENAI_API_KEY")

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
import sys

VOICE_PATH = '/home/rokey/ros2_ws/Tutorial/VoiceProcessing'
sys.path.append(VOICE_PATH)


from dotenv import load_dotenv
env_path = os.path.join(VOICE_PATH, '.env')
load_dotenv(dotenv_path=env_path)


# =========================================================
# [추가] 음성/AI 관련 라이브러리
# =========================================================
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from STT import STT

# .env 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# =========================================================

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# =========================================================
# 2. 음성 지능 클래스 (Voice Brain)
# =========================================================
class VoiceBrain:
    def __init__(self, api_key):
        if not api_key:
            print("❌ [Error] OPENAI_API_KEY가 없습니다.")
            return
        
        self.stt = STT(api_key)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=api_key)
        
        self.prompt = PromptTemplate.from_template(
            """
            당신은 로봇의 귀입니다. 사용자의 한국어 명령을 듣고, 
            집어야 할 '물건의 이름(영어 소문자)' 하나만 정확히 출력하세요.

            <규칙>
            1. 문장에서 잡아야 할 물건(명사)을 추출하여 영어로 번역할 것.
            2. 잡담이나 설명 없이 오직 '영어단어' 하나만 출력할 것.
            3. 잡을 물건이 없으면 'none' 출력.

            <예시>
            "사과 집어줘" -> apple
            "저기 있는 레몬 줘" -> lemon
            "컵 옮겨" -> cup
            "안녕" -> none

            <사용자 입력>
            "{user_input}"
            """
        )
        self.chain = self.prompt | self.llm

    def listen_and_get_target(self):
        print("\n🎤 [Listening] 말씀해 주세요... (녹음 시작)")
        user_text = self.stt.speech2text()
        print(f"📝 [STT] 인식된 문장: \"{user_text}\"")
        
        if not user_text: return None

        response = self.chain.invoke({"user_input": user_text})
        target_keyword = response.content.strip().lower()
        
        print(f"🧠 [LLM] 추출된 타겟: [{target_keyword}]")
        
        if "none" in target_keyword: return None
        return target_keyword

# =========================================================
# 3. 로봇 자동화 클래스 (OBB 적용 + 회전 최적화)
# =========================================================
class AutomationNode(Node):
    def __init__(self):
        super().__init__("automation_node")
        
        cv2.namedWindow("Webcam")
        self.img_node = ImgNode()
        
        # OBB 모델 로드 확인
        try:
            self.model = YOLO("best.pt") # 반드시 OBB로 학습된 모델이어야 함
            self.get_logger().info(f"✅ YOLO Loaded! Classes: {self.model.names}")
        except Exception as e:
            self.get_logger().error(f"YOLO Load Failed: {e}")

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
            self.gripper2cam = np.eye(4)
        
    def normalize_angle(self, angle_deg):
        """
        [최적화 로직] 각도를 -180 ~ 180도 사이로 변환
        예: 200도 -> -160도
        """
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg <= -180:
            angle_deg += 360
        return angle_deg

    def detect_and_pick(self, target_name):
        color_frame = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()
        
        TARGET_CLASS = target_name.lower().strip()

        if color_frame is None or depth_frame is None:
            self.get_logger().warn("No frame received")
            return

        self.get_logger().info(f"🔍 Searching for target: '{TARGET_CLASS}'...")
        
        # OBB 예측 실행
        results = self.model.predict(color_frame, conf=0.5, verbose=False)
        
        # OBB 결과가 있는지 확인 (없으면 일반 박스 boxes 사용)
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            detections = results[0].obb
            is_obb = True
        else:
            detections = results[0].boxes
            is_obb = False
            self.get_logger().warn("⚠️ 일반 박스 모델입니다. 회전 감지 불가.")

        if len(detections) == 0:
            self.get_logger().warn("No objects detected.")
            return

        target_data = None
        found_target = False

        print("-" * 30)
        # 클래스 매칭
        # OBB의 경우 xywhr, 일반 박스의 경우 xywh
        for i, cls_tensor in enumerate(detections.cls):
            class_id = int(cls_tensor.item())
            class_name = results[0].names[class_id]
            conf = detections.conf[i].item()
            print(f"👀 Scan: '{class_name}' ({conf:.2f})")

            if class_name == TARGET_CLASS:
                if is_obb:
                    # x, y, w, h, rotation(rad)
                    target_data = detections.xywhr[i].cpu().numpy()
                else:
                    # x, y, w, h
                    x, y, w, h = detections.xywh[i].cpu().numpy()
                    target_data = [x, y, w, h, 0.0] # 회전 0도 가정
                
                found_target = True
                print(f"   ✅ Target Match! Found '{TARGET_CLASS}'")
                break 

        print("-" * 30)

        if not found_target:
            self.get_logger().warn(f"🚫 화면에 '{TARGET_CLASS}'가 없습니다!")
            return

        # 좌표 및 각도 추출
        cx, cy, w, h, rot_rad = target_data
        x, y = int(cx), int(cy)
        
        # 라디안 -> 도 변환
        angle_deg = np.degrees(rot_rad)

        # [각도 보정] 긴 쪽(Height)이 아니라 짧은 쪽(Width)을 잡아야 한다면 90도 회전
        # (그리퍼 형상에 따라 선택)
        if w < h:
            angle_deg += 90
            
        # [각도 최적화] 200 -> -160 변환 적용
        optimized_angle = self.normalize_angle(angle_deg)
        
        print(f"🎯 Target: {TARGET_CLASS}")
        print(f"   📍 Pixel: ({x}, {y})")
        print(f"   🔄 Original Angle: {np.degrees(rot_rad):.2f}")
        print(f"   ⚡ Optimized Angle: {optimized_angle:.2f} (To Robot RZ)")

        z = self.get_depth_value(x, y, depth_frame)
        if z is None or z == 0:
            self.get_logger().warn("Depth invalid, skipping...")
            return

        camera_pos = self.get_camera_pos(x, y, z, self.intrinsics)
        robot_pos = self.transform_to_base(camera_pos)
        
        print(f"   📷 Camera Coord: {camera_pos}")
        print(f"   🤖 Robot Base Coord: {robot_pos}")

        # 시각화
        cv2.circle(color_frame, (x, y), 5, (0, 0, 255), -1)
        try:
            annotated = results[0].plot() # OBB 박스 그리기
            cv2.imshow("Webcam", annotated)
        except:
            cv2.imshow("Webcam", color_frame)
        cv2.waitKey(100)

        # 로봇 실행 (최적화된 각도 전달)
        self.execute_pick_sequence(robot_pos, optimized_angle)

    def execute_pick_sequence(self, robot_coordinate, target_rz):
        x, y, z = robot_coordinate
        
        # 1. 초기 위치로 이동 (케이블 꼬임 방지 1차)
        movej(self.JReady, vel=VELOCITY, acc=ACC)
        
        # 현재 자세(RX, RY) 가져오기 -> 수직 자세 유지
        curr = get_current_posx()[0]
        # target_rz 적용
        
        # 2. 이동 좌표 생성
        # [사용자 튜닝값] Z+170(Grip), Z+300(Approach)
        grip_pos = posx([x, y, z + 170, curr[3], curr[4], target_rz])
        app_pos = posx([x, y, z + 300, curr[3], curr[4], target_rz])
        
        self.get_logger().info(f"📍 Approach with RZ: {target_rz:.2f}")

        # 3. 접근 및 피킹
        self.gripper.open_gripper() # Open
        wait(1)
        
        movel(app_pos, vel=VELOCITY, acc=ACC) # 접근
        wait(0.5)
        
        movel(grip_pos, vel=VELOCITY, acc=ACC) # 하강 (회전 적용됨)
        wait(0.5)
        
        self.gripper.close_gripper(150) # 부드럽게 잡기
        wait(1)
        
        # 4. 상승 및 놓기
        movel([0, 0, 150, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
        wait(1)
        
        # (테스트용) 제자리 놓기 -> 실제로는 바구니 위치로 이동하면 됨
        movel(grip_pos, vel=VELOCITY, acc=ACC)
        self.gripper.open_gripper()
        wait(1)
        
        movel(app_pos, vel=VELOCITY, acc=ACC)
        
        # 5. [중요] 초기 자세 복귀 (Cable Unwinding)
        # movej를 사용하여 절대 관절 각도(JReady)로 돌아갑니다.
        # 이 과정에서 돌아갔던 손목이 반대로 풀리며 원상복구 됩니다.
        print("🔄 Returning to Home (Unwinding Cables)...")
        movej(self.JReady, vel=VELOCITY, acc=ACC)
        
        print("✅ Task Finished. Ready for next input.")

    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z
        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=[x,y,z]
        return T

    def transform_to_base(self, camera_coords):
        coord = np.append(np.array(camera_coords), 1)
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        
        # [사용자 튜닝 값]
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
        if len(valid_depths) == 0: return None
        return np.median(valid_depths)

    def update_view(self):
        rclpy.spin_once(self.img_node, timeout_sec=0.001)
        img = self.img_node.get_color_frame()
        if img is not None:
            cv2.imshow("Webcam", img) 
        return img

# =========================================================
# Main Loop
# =========================================================
if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_voice_obb_automation", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait, DR_MV_MOD_REL
        from DR_common2 import posx, posj
    except ImportError: exit(True)

    # 인스턴스 생성
    voice_brain = VoiceBrain(openai_api_key)
    auto_node = AutomationNode()

    print("🚀 Initializing Robot Position to JReady...")
    movej(auto_node.JReady, vel=VELOCITY, acc=ACC)
    print("✅ Initialization Complete!")

    print("\n" + "="*60)
    print("   [Spacebar]: 🎤 음성 명령 듣기 ('오이 집어줘')")
    print("   [Enter]: ⌨️ 텍스트 입력 모드")
    print("   [ESC]: 종료")
    print("="*60 + "\n")

    while rclpy.ok():
        auto_node.update_view()
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # Spacebar
            target_obj = voice_brain.listen_and_get_target()
            if target_obj:
                print(f"🚀 [음성 명령] '{target_obj}' 집으러 갑니다!")
                auto_node.detect_and_pick(target_obj)
            else:
                print("⚠️ 타겟 없음.")

        elif key == 13: # Enter
            print("\n" + "-"*40)
            target_input = input(">> 찾을 물건 입력: ")
            if target_input.strip():
                auto_node.detect_and_pick(target_input)

        elif key == 27:
            break

    cv2.destroyAllWindows()
    auto_node.destroy_node()
    rclpy.shutdown()