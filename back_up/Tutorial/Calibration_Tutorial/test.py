import cv2
import rclpy
from rclpy.node import Node
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG

import time
import numpy as np
import DR_init

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


# 마우스 콜백 함수
class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")
        
        cv2.namedWindow("Webcam")

        self.img_node = ImgNode()
        
        # 카메라 내부 파라미터(Intrinsics)가 수신될 때까지 대기하는 루프
        self.intrinsics = None
        # 초기 위치 및 복구 자세(JReady)
        self.JReady = posj([-3.98, 7.54, 64.19, -6.33, 106.34, -3.48])
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

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

        try:
            self.gripper2cam = np.load("T_gripper2camera.npy")
        except FileNotFoundError:
            self.get_logger().error("T_gripper2camera.npy not found. Using identity matrix.")
            self.gripper2cam = np.eye(4)

        # 마우스 콜백 설정
        cv2.setMouseCallback("Webcam", self.mouse_callback)
        self.get_logger().info("마우스 콜백 설정 완료. 화면을 클릭하세요.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_frame = self.img_node.get_depth_frame()
            
            attempts = 0
            while (depth_frame is None or np.all(depth_frame == 0)) and attempts < 10:
                self.get_logger().info("Retry get depth img...")
                rclpy.spin_once(self.img_node)
                depth_frame = self.img_node.get_depth_frame()
                attempts += 1
                time.sleep(0.1)

            if depth_frame is None:
                return

            print(f"img cordinate: ({x}, {y})")
            z = self.get_depth_value(x, y, depth_frame)
            
            if z is None:
                self.get_logger().warn("Depth invalid, using default Z=500.0")
                z = 500.0

            camera_center_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            print(f"camera cordinate: ({camera_center_pos})")

            robot_coordinate = self.transform_to_base(camera_center_pos)
            print(f"robot cordinate: ({robot_coordinate})")

            # --- 시작 시 원점 ---
            self.get_logger().info("Init Home (JReady)")
            movej(self.JReady, vel=VELOCITY, acc=ACC)
            print("=" * 100)

            # --- X, Y 이동 확인 로직 호출 ---
            self.pick_and_drop(*robot_coordinate)
            
            # --- 동작 완료 후 원점 복귀 ---
            self.get_logger().info("Return to Home (JReady)")
            movej(self.JReady, vel=VELOCITY, acc=ACC)
            print("=" * 100)

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

    def pick_and_drop(self, x, y, z):

        current_pos = get_current_posx()[0]
        
        self.gripper.move_gripper(700)  #그리퍼 열기
        wait(1)

        target_pos1 = posx([x, y, z + 200, current_pos[3], current_pos[4], current_pos[5]])
        
        self.get_logger().info(f"Moving to Target (X, Y): {target_pos1}")

        # (1) 목표 위치로 이동 (Z축은 SAFE_Z 유지, 그리퍼 동작 없음)
        movel(target_pos1, vel=VELOCITY, acc=ACC)
        wait(1) 

        self.gripper.close_gripper()    #그리퍼 닫기
        wait(1)
        self.get_logger().info(f"gripped object")

        movel([0,0,200,0,0,0], vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)    #상승
        wait(1)

        movel(target_pos1, vel=VELOCITY, acc=ACC)  #반납(쿠션 위 기준)
        wait(1)

        self.gripper.move_gripper(700)  #그리퍼 다시 열기(open 사용해도 됨)
        wait(1)



    def transform_to_base(self, camera_coords):
        coord = np.append(np.array(camera_coords), 1)
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def update_image(self):
        rclpy.spin_once(self.img_node, timeout_sec=0.001)
        img = self.img_node.get_color_frame()
        if img is not None:
            cv2.imshow("Webcam", img)
        return img

    def get_depth_value(self, center_x, center_y, depth_frame):
        height, width = depth_frame.shape
        if 0 <= center_x < width and 0 <= center_y < height:
            depth_value = depth_frame[center_y, center_x]
            if depth_value > 0:
                return depth_value
        return None


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)
    
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import (
            get_current_posx,
            movej,
            movel,
            wait,
            DR_MV_MOD_REL
        )
        from DR_common2 import posx, posj

    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        exit(True)

    test_node = TestNode()

    while rclpy.ok():
        test_node.update_image()
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()
    test_node.destroy_node()
    rclpy.shutdown()