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

# ---------------------------
# 로봇 기본 설정
# ---------------------------
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"


class YoloRobotNode(Node):
    def __init__(self):
        super().__init__("yolo_pick_node")

        # 1. RealSense 노드 생성
        self.img_node = ImgNode()
        rclpy.spin_once(self.img_node)
        time.sleep(1)

        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.gripper2cam = np.load("T_gripper2camera.npy")

        # 홈 위치
        self.JReady = posj([0, 0, 90, 0, 90, -90])
        self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        # NOTE: Segmentation 모델로 변경 (자동 다운로드됨)
        print("Loading YOLO Segmentation model...")
        self.model = YOLO("yolo11n-seg.pt") 
        self.latest_results = None

        self.is_moving = False

        cv2.setMouseCallback("YOLO Robot View", self.mouse_callback)

    # ==============================================================
    #  [NEW] Segmentation 마스크 기반 각도 계산
    # ==============================================================
    def get_angle_from_segmentation(self, mask_contour):
        """
        YOLO가 제공한 Segmentation 윤곽선(contour)을 이용해
        '짧은 면'을 잡기 위한 회전 각도를 정밀하게 계산합니다.
        """
        # YOLO 윤곽선 데이터를 OpenCV 형식으로 변환 (Nx2 int array)
        cnt = np.array(mask_contour, dtype=np.int32)

        if len(cnt) == 0:
            return 0.0

        # 최소 면적 사각형 (Rotated Rectangle) 계산
        # 이 방식은 마스크의 정확한 모양을 기반으로 하므로 이전보다 훨씬 정확합니다.
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        # 짧은 면을 잡도록 각도 보정 로직 (이전과 동일)
        if rw < rh:
            target_angle = angle
        else:
            target_angle = angle + 90

        print(f"[Seg Angle] W:{rw:.1f}, H:{rh:.1f}, Org:{angle:.1f} -> Target:{target_angle:.1f}")
        return target_angle

    # ==============================================================
    #  마우스 클릭 콜백 (Segmentation 데이터 활용)
    # ==============================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_moving:
                print("⚠️ Robot is busy")
                return
            
            if self.latest_results is None or self.latest_results[0].masks is None:
                 print("⚠️ No segmentation results available.")
                 return

            # 1. 클릭한 지점이 어떤 객체의 바운딩 박스 안에 있는지 확인
            selected_index = -1
            boxes = self.latest_results[0].boxes
            for i, box in enumerate(boxes):
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    selected_index = i
                    # 겹쳐있을 경우, 가장 신뢰도 높은 첫 번째 것을 선택하고 종료
                    break
            
            if selected_index == -1:
                print("⚠️ No object detected at click point.")
                return

            # 2. 선택된 객체의 Segmentation 윤곽선 가져오기
            # masks.xy는 각 객체의 윤곽선 포인트 리스트를 담고 있음
            target_contour = self.latest_results[0].masks.xy[selected_index]

            depth_frame = self.img_node.get_depth_frame()

            # 3. Depth(Z) 값 획득
            z = self.get_depth_value(x, y, depth_frame)
            if z is None or z == 0:
                print("⚠️ No depth info")
                return

            # 4. [NEW] Segmentation 기반 각도 계산
            angle_offset = self.get_angle_from_segmentation(target_contour)

            # 5. 좌표 변환
            cam_pos = self.get_camera_pos(x, y, z, self.intrinsics)
            robot_coord = self.transform_to_base(cam_pos)

            print(f"[Action] Pick Seg Object! Angle Offset: {angle_offset:.1f}")
            self.execute_pick_and_drop(robot_coord, angle_offset)

    # ==============================================================
    #  YOLO Segmentation 감지 및 표시
    # ==============================================================
    def process_frame(self):
        rclpy.spin_once(self.img_node)
        color_img = self.img_node.get_color_frame()
        
        if color_img is None: return

        # Segmentation 모델 예측 실행 (conf 낮춤 유지)
        self.latest_results = self.model.predict(color_img, conf=0.25, verbose=False)
        
        # plot()은 Segmentation 마스크가 있으면 자동으로 색칠해서 보여줍니다.
        annotated = self.latest_results[0].plot()

        cv2.imshow("YOLO Robot View", annotated)

    # ==============================================================
    #  Pick & Place 실행 래퍼
    # ==============================================================
    def execute_pick_and_drop(self, robot_coord, angle_offset):
        self.is_moving = True
        try:
            x, y, z = robot_coord
            self.pick_and_drop(x, y, z, angle_offset)
        finally:
            self.is_moving = False

    # ==============================================================
    #  로봇 동작 (각도 유지)
    # ==============================================================
    def pick_and_drop(self, x, y, z, angle_offset):
        Z_OFFSET = 170
        z += Z_OFFSET

        current = get_current_posx()[0]
        curr_rx, curr_ry, curr_rz = current[3], current[4], current[5]

        # 목표 회전 각도 계산
        target_rz = curr_rz + angle_offset

        approach_h = 50

        # 모든 이동 단계에서 계산된 target_rz 유지
        approach = posx([x, y, z + approach_h, curr_rx, curr_ry, target_rz])
        pick     = posx([x, y, z,              curr_rx, curr_ry, target_rz])
        lift     = posx([x, y, z + approach_h, curr_rx, curr_ry, target_rz])
        drop     = posx([x + 200, y, z + approach_h, curr_rx, curr_ry, target_rz])

        print(f"➡ Approach (RZ: {target_rz:.2f})")
        movel(approach, vel=50, acc=50); wait(0.5)

        print("➡ Pick")
        movel(pick, vel=30, acc=30); wait(0.3)
        self.gripper.close_gripper(); wait(1.0)

        print("➡ Lift")
        movel(lift, vel=50, acc=50); wait(0.3)

        print("➡ Drop (Angle Maintained)")
        movel(drop, vel=50, acc=50); wait(0.3)
        self.gripper.open_gripper(); wait(1.0)

        print("➡ Return")
        movej(self.JReady, vel=60, acc=60); wait(0.5)
        print("✓ Done")

    # ==============================================================
    #  유틸리티
    # ==============================================================
    def get_camera_pos(self, px, py, pz, K):
        cx = (px - K["ppx"]) * pz / K["fx"]
        cy = (py - K["ppy"]) * pz / K["fy"]
        return (cx, cy, pz)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords):
        coord = np.append(camera_coords, 1)
        base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        base2cam = base2gripper @ self.gripper2cam
        td = base2cam @ coord
        return td[:3]

    def get_depth_value(self, x, y, depth_frame):
        h, w = depth_frame.shape
        if 0 <= x < w and 0 <= y < h:
            return depth_frame[y, x]
        return None


# ==============================================================
#  Main
# ==============================================================
if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_yolo_demo", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import get_current_posx, movej, movel, wait
    from DR_common2 import posx, posj

    cv2.namedWindow("YOLO Robot View")
    yolo_node = YoloRobotNode()

    try:
        while True:
            yolo_node.process_frame()
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()