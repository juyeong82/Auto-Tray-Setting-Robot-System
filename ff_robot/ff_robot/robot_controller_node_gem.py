#!/usr/bin/env python3
# robot_controller_node_gem.py (Strict Verification Version)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import numpy as np
import os
import sys # 종료를 위해 추가
from scipy.spatial.transform import Rotation as R

import DR_init
from ff_robot_interfaces.srv import OrderService, DetectObject
from ff_robot.order_logic import SlotManager

# --- [설정] 안전 검증 높이 ---
SAFE_Z_HEIGHT = 400.0  # (mm)

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# [설정] 관측 자세
J_LOOK_POS = [-42.94, -54.85, 50.51, -3.11, 130.60, -266.60]

# [설정] 임시 Place 위치 (Loop 연결용)
# x=300, y=10, z=400, rx=0, ry=180, rz=0
TEMP_PLACE_POS = [300.0, 10.0, 400.0, 0.0, 180.0, 0.0]

# --- 전역 변수 ---
node_ = None          
dsr_control_node_ = None 
manager = None
vision_cli = None 
gripper = None
T_GRIPPER_TO_CAM = None

# ==============================================================================
# [설정 1] 캘리브레이션 파일 로드 (엄격 모드)
# ==============================================================================
def load_calibration():
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    
    if os.path.exists(npy_path):
        T_GRIPPER_TO_CAM = np.load(npy_path)
        print(f"✅ 캘리브레이션 파일 로드 성공: {npy_path}")
        print(T_GRIPPER_TO_CAM)
    else:
        print(f"\n❌ [CRITICAL ERROR] 캘리브레이션 파일이 없습니다!")
        print(f"   경로: {npy_path}")
        print("   프로그램을 종료합니다.\n")
        sys.exit(1) # 강제 종료

# ==============================================================================
# [설정 2] 가상 그리퍼 강제 사용
# ==============================================================================
class RG:
    def __init__(self, *args): 
        print("✅ [Gripper] 가상(Dummy) 그리퍼 모드로 동작합니다.")
    def open_gripper(self): 
        print("   👐 [Virtual] Gripper Open")
    def close_gripper(self, force=None): 
        print("   ✊ [Virtual] Gripper Close")
    def get_status(self):
        return "Virtual OK"

# ==============================================================================
# [Math] 좌표 변환
# ==============================================================================
def get_robot_pose_matrix(posx_list):
    x, y, z, rx, ry, rz = posx_list
    rot = R.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T

def transform_camera_to_base(cam_xyz):
    from DSR_ROBOT2 import get_current_posx
    
    # 1. 현재 로봇 끝단 위치
    curr_posx = get_current_posx()[0]
    T_base_gripper = get_robot_pose_matrix(curr_posx)
    
    # 2. Eye-in-Hand 변환
    T_base_cam = T_base_gripper @ T_GRIPPER_TO_CAM
    
    # 3. 물체 좌표 (Camera -> Base)
    # yolo_vision_node가 미터(m) 단위로 준다고 가정 -> 1000 곱해서 mm로 변환
    p_cam = np.array([cam_xyz[0]*1000, cam_xyz[1]*1000, cam_xyz[2]*1000, 1.0])
    
    p_base = T_base_cam @ p_cam
    return p_base[:3]

# ==============================================================================
# [Motion]
# ==============================================================================
def wait_for_motion():
    from DSR_ROBOT2 import check_motion
    time.sleep(0.2)
    while check_motion() != 0:
        time.sleep(0.1) 
        if not rclpy.ok(): return False
    return True

def safe_movel(pos, desc="이동"):
    global node_
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    
    p_str = f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
    node_.get_logger().info(f"   🏃 {desc}... 목표: {p_str}")
    
    try:
        time.sleep(1.0) 
        # 속도를 낮춰서 안전하게 이동 (Vel=40)
        movel(pos, vel=[40, 40], acc=[40, 40], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.5) 
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

def safe_move_and_check(target_pos):
    global gripper, node_
    
    # 1. Z값을 SAFE_Z_HEIGHT로 고정 (안전장치)
    safe_target = list(target_pos[:3])
    safe_target[2] = SAFE_Z_HEIGHT 
    
    # 2. 바닥 보는 자세 [0, 180, 0] 적용
    final_pose = safe_target + [0.0, 180.0, 0.0]
    
    node_.get_logger().info(f"   🛡️ Pick 검증 이동: Z={SAFE_Z_HEIGHT}mm 고정")

    # 3. 이동 (Hovering)
    if not safe_movel(final_pose, "Pick 위치 상공으로 이동"): return False
    
    # 4. 도착 확인 (그리퍼 깜빡임)
    if gripper: 
        node_.get_logger().info("   👀 도착! (그리퍼 동작 확인)")
        gripper.open_gripper()
        time.sleep(0.5)
        gripper.close_gripper()
        time.sleep(0.5)

    return True

# ==============================================================================
# [Task]
# ==============================================================================
def call_vision_service(target_name):
    global node_, vision_cli
    if vision_cli is None or not vision_cli.service_is_ready(): return None
    
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 3.0: return None
        time.sleep(0.01)
    try:
        res = future.result()
        if res.found: return [res.position.x, res.position.y, res.position.z]
    except: pass
    return None

def perform_robot_task():
    global node_, manager, gripper
    
    try: from DSR_ROBOT2 import movej
    except: return

    # 가상 그리퍼 사용
    gripper = RG() 
    
    node_.get_logger().info("[Task] 로봇 준비 완료. 주문 대기...")

    while rclpy.ok():
        try:
            needed_items = manager.get_all_needed_items()
            if not needed_items:
                time.sleep(1.0)
                continue

            target_item = needed_items[0]
            node_.get_logger().info(f"========================================")
            node_.get_logger().info(f"🔎 [Search] '{target_item}' 찾는 중...")

            # 1. 관측 위치 이동 (매번 이동해서 정확하게 봄)
            node_.get_logger().info(f"🔭 관측 위치 이동...")
            movej(J_LOOK_POS, vel=40, acc=40)
            wait_for_motion()

            # 2. 비전 좌표
            cam_xyz = call_vision_service(target_item)
            if cam_xyz is None:
                node_.get_logger().warn("   ⚠️ 타겟 못 찾음 (Retry)")
                time.sleep(1.0)
                continue

            # 3. 좌표 변환
            base_xyz = transform_camera_to_base(cam_xyz)
            node_.get_logger().info(f"   📊 [Debug] 변환 좌표: X={base_xyz[0]:.1f}, Y={base_xyz[1]:.1f}")

            # 4. Pick 검증 이동 (Z=400 고정)
            if safe_move_and_check(base_xyz):
                node_.get_logger().info("✅ Pick 위치 확인 완료.")
                
                # 5. Place (고정 위치로 이동)
                node_.get_logger().info(f"🚚 Place (Test) 이동 중... {TEMP_PLACE_POS[:3]}")
                if safe_movel(TEMP_PLACE_POS, "트레이(임시) 이동"):
                    # 도착 후 그리퍼 열기
                    gripper.open_gripper()
                    time.sleep(0.5)
                    
                    # [중요] 완료 처리 -> 다음 아이템으로 넘어감
                    if "SWAP" in "PLACE" or "FINISH" in "PLACE": # 임시 조건
                        pass
                    
                    # 로직상 이 아이템은 처리된 것으로 간주
                    # manager의 상태를 강제로 업데이트해줘야 함 (SlotManager 로직에 따름)
                    # 여기서는 간단히 clear_slot 호출 (0번 슬롯이 비워지면 다음 주문 처리)
                    # 실제로는 placed_total 카운트를 올려야 하지만, 테스트니 슬롯을 비움
                    manager.clear_slot(0) 
                    node_.get_logger().info("🎉 작업 1회 완료. 다음 작업 준비.")
            else:
                node_.get_logger().error("❌ Pick 이동 실패")

            time.sleep(1.0)

        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

# ==============================================================================
# [Main]
# ==============================================================================
def handle_order_request(request, response):
    global manager, node_
    node_.get_logger().info(f"⚡ [Service] 주문: {request.item_names}")
    success, msg = manager.check_and_deduct_stock(request.item_names, request.item_quantities)
    if success:
        manager.add_order_to_slot(f"ORD-{int(time.time())}", request.item_names, request.item_quantities)
        response.success = True
        response.message = "접수 완료"
    return response

def main(args=None):
    global node_, dsr_control_node_, manager, vision_cli, T_GRIPPER_TO_CAM
    rclpy.init(args=args)
    
    # 캘리브레이션 파일 먼저 체크
    load_calibration()
    
    manager = SlotManager()
    
    node_ = rclpy.create_node("robot_controller_node", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_ 

    cb_group = ReentrantCallbackGroup()
    node_.create_service(OrderService, '/dsr01/order_service', handle_order_request, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)

    executor = MultiThreadedExecutor()
    executor.add_node(node_)

    t_robot = threading.Thread(target=perform_robot_task, daemon=True)
    t_robot.start()
    
    try: executor.spin()
    except: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()