import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from onrobot import RG
from ultralytics import YOLO
import time
import numpy as np
import DR_init
import sys

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


# [TODO] 사용자가 정의한 서비스 인터페이스를 import 하세요.
from my_package.srv import OrderService

class MainControl(Node):
    def __init__(self):
        super().__init__("main_control_node")

        # 1. 서비스 서버 생성
        # 'process_order'는 서비스 이름이므로 메인 노드와 맞춰야 합니다.
        self.srv = self.create_service(
            OrderService, 
            'process_order', 
            self.order_callback
        )
        
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

    def order_callback(self, request, response):

        target_list = request.target_list
        self.get_logger().info(f"📥 요청 수신: {target_list}")
        
        all_tasks_success = True

        # 요청받은 리스트를 순회하며 작업 수행
        for target_name in target_list:
            # 개별 물체 처리 로직 호출
            result = self.execute_pick_and_place(target_name)
            
            if not result:
                self.get_logger().warn(f"❌ 실패: {target_name}")
                all_tasks_success = False
                # 실패 시 루프를 멈출지 계속할지는 정책에 따라 결정 (여기선 break)
                break 
        
        # 결과 응답 설정
        response.success = all_tasks_success
        return response

    def execute_pick(self, target_name):
        """
        단일 물체에 대한 감지 -> 좌표변환 -> 파지 -> 배치 로직
        성공 시 True, 실패 시 False 반환
        """
        # 1. 관측 위치(JReady)로 이동
        adsjlk;faj
        # 3. 좌표 변환 (Pixel -> Camera -> Robot Base)
        
        # 4. 로봇 모션 실행 (접근 -> 파지 -> 들어올리기 -> 트레이 배치 -> 복귀)
        
        return True

if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_voice_obb_automation", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait, DR_MV_MOD_REL
        from DR_common2 import posx, posj
    except ImportError: exit(True)

    auto_node = AutomationNode()