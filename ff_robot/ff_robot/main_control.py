import cv2
import rclpy
import json
import time
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String

# --- 사용자 라이브러리 (기존 코드 유지) ---
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG
from ultralytics import YOLO
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
        super().__init__("dsr_vision_node")
        
        # --- 1. 통신 설정 (Main Node와 연동) ---
        # [Sub] 메인 노드로부터 타겟 리스트 수신 (예: '["apple", "banana"]')
        self.target_sub = self.create_subscription(
            String, 
            '/vision_target_list', 
            self.target_list_callback, 
            10
        )
        
        # [Pub] 작업 완료 상태 전송
        self.status_pub = self.create_publisher(String, '/robot_action_status', 10)

        # 내부 작업 큐 (메인에서 받은 리스트를 저장해두고 하나씩 처리)
        self.work_queue = []
        self.is_busy = False # 현재 로봇이 움직이는 중인지 확인

        # --- 2. 하드웨어 및 모델 초기화 ---
        cv2.namedWindow("Webcam")
        self.img_node = ImgNode()
        
        try:
            self.model = YOLO("best.pt") 
            self.get_logger().info("✅ YOLO Model Loaded Successfully!")
        except Exception as e:
            self.get_logger().error(f"YOLO Model Load Failed: {e}")

        self.intrinsics = None
        # 초기 관측 자세 (Look Pose)
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
        
        self.get_logger().info("🚀 Vision Node Ready. Waiting for orders...")

    def target_list_callback(self, msg):
        """
        메인 노드에서 주문 리스트가 들어오면 큐에 추가합니다.
        msg.data 예시: '["juice", "coffee"]'
        """
        try:
            new_targets = json.loads(msg.data)
            if isinstance(new_targets, list):
                self.work_queue.extend(new_targets)
                self.get_logger().info(f"📥 New Orders Received: {new_targets}. Current Queue: {self.work_queue}")
            else:
                self.get_logger().warn("Invalid format: Not a list")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode JSON message")

    def process_queue(self):
        """
        메인 루프에서 호출됨. 큐에 작업이 있고 로봇이 쉬고 있다면 작업을 시작함.
        """
        if not self.is_busy and self.work_queue:
            target = self.work_queue.pop(0) # FIFO: 맨 앞의 주문 꺼내기
            self.is_busy = True
            
            self.get_logger().info(f"▶️ Starting Task for: {target}")
            
            # 1. 관측 위치로 이동
            movej(self.JReady, vel=VELOCITY, acc=ACC)
            wait(0.5) # 카메라 안정화 대기

            # 2. 감지 및 파지 시도
            success = self.detect_and_pick(target)
            
            # 3. 결과 보고
            status_msg = String()
            if success:
                status_msg.data = "pick_done"
                self.get_logger().info(f"✅ Task Completed for {target}")
            else:
                status_msg.data = "fail"
                self.get_logger().warn(f"❌ Task Failed for {target}")
                # 실패한 경우 재시도를 위해 큐에 다시 넣을지, 그냥 넘길지는 정책 결정 필요
                # 여기서는 그냥 넘기는 것으로 처리
            
            self.status_pub.publish(status_msg)
            self.is_busy = False

    def detect_and_pick(self, target_name):
        """
        리턴값: 성공 시 True, 실패 시 False
        """
        # 최신 프레임을 가져오기 위해 잠깐 스핀
        rclpy.spin_once(self.img_node, timeout_sec=0.1)
        
        color_frame = self.img_node.get_color_frame()
        depth_frame = self.img_node.get_depth_frame()
        TARGET_CLASS = target_name.lower().strip()

        if color_frame is None or depth_frame is None:
            self.get_logger().warn("No frame received")
            return False

        # YOLO 예측
        results = self.model.predict(color_frame, conf=0.5, verbose=False)
        
        if len(results[0].boxes) == 0:
            self.get_logger().warn("No objects detected.")
            return False

        target_box = None
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = results[0].names[class_id]
            if class_name == TARGET_CLASS:
                target_box = box
                break 

        if target_box is None:
            self.get_logger().warn(f"🚫 Target '{TARGET_CLASS}' not found in view.")
            return False

        # 좌표 계산
        x_center, y_center, w, h = target_box.xywh[0].cpu().numpy()
        x, y = int(x_center), int(y_center)
        z = self.get_depth_value(x, y, depth_frame)
        
        if z is None or z == 0:
            self.get_logger().warn("Invalid Depth.")
            return False

        # 로봇 좌표계 변환
        camera_pos = self.get_camera_pos(x, y, z, self.intrinsics)
        robot_pos = self.transform_to_base(camera_pos)
        
        # 시각화
        cv2.circle(color_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(color_frame, f"Pick: {TARGET_CLASS}", (x, y-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Webcam", color_frame)
        cv2.waitKey(1) # 화면 갱신

        # 로봇 파지 동작 실행
        self.execute_pick_sequence(robot_pos)
        return True

    def execute_pick_sequence(self, robot_coordinate):
        x, y, z = robot_coordinate
        # 파지 및 배치 (Pick -> Place)
        self.pick_and_drop(x, y, z)
        # 복귀
        movej(self.JReady, vel=VELOCITY, acc=ACC)

    # --- (기존 좌표계산 함수들: get_camera_pos, transform_to_base 등은 그대로 유지) ---
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
        
        # [옵션] 좌표 보정
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

    def pick_and_drop(self, x, y, z):
        current_pos = get_current_posx()[0]
        self.gripper.open_gripper()
        wait(1)

        # 위치 설정
        target_grip_pos = posx([x, y, z + 170, current_pos[3], current_pos[4], current_pos[5]])
        target_approach_pos = posx([x, y, z + 300, current_pos[3], current_pos[4], current_pos[5]])
        
        # 1. 접근 및 파지
        movel(target_approach_pos, vel=VELOCITY, acc=ACC) 
        wait(0.5)
        movel(target_grip_pos, vel=VELOCITY, acc=ACC) 
        wait(0.5)
        self.gripper.close_gripper(150) 
        wait(1)
        
        # 2. 들어올리기
        movel([0, 0, 150, 0, 0, 0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL) 
        wait(1)
        
        # 3. 트레이 배치 (여기서는 예시로 제자리 놓기로 되어있으나, 서빙 트레이 좌표로 수정 필요)
        # 트레이 좌표 예시: tray_pos = posx([...])
        # movel(tray_pos, ...)
        
        movel(target_grip_pos, vel=VELOCITY, acc=ACC) # (테스트: 제자리 놓기)
        self.gripper.open_gripper()
        wait(1)
        
        movel(target_approach_pos, vel=VELOCITY, acc=ACC) 

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

    # DSR 라이브러리 임포트 (이 부분이 반드시 실행되어야 로봇 제어가 가능)
    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait, DR_MV_MOD_REL
        from DR_common2 import posx, posj
    except ImportError:
        print("Error: DSR Library not found!")
        exit(True)

    auto_node = AutomationNode()

    # 초기 위치 이동
    print("🚀 Moving to Initial Position...")
    movej(auto_node.JReady, vel=VELOCITY, acc=ACC)
    
    print("\n✅ System Ready. Waiting for ROS Topic commands...")

    # [핵심] 메인 실행 루프
    while rclpy.ok():
        # 1. ROS 콜백 처리 (주문 수신용)
        rclpy.spin_once(auto_node, timeout_sec=0.01)
        
        # 2. 카메라 화면 갱신
        auto_node.update_view()
        cv2.waitKey(1)

        # 3. 작업 큐 확인 및 로봇 동작 실행
        # (토픽으로 받은 주문이 큐에 쌓여있으면 순차적으로 실행)
        auto_node.process_queue()

    cv2.destroyAllWindows()
    auto_node.destroy_node()
    rclpy.shutdown()