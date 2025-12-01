#!/usr/bin/env python3
# robot_controller_node.py (Threaded Fix + GripperManager + 2-step Pick)
# MISSION 
# 음성 매끄럽게!!!
# 키오스크 주문 리스트 수량 확인!!!

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

import DR_init
from ff_robot_interfaces.srv import OrderService, DetectObject
from ff_robot.order_logic import SlotManager
from ff_robot.gripper import GripperManager 

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# [🎛️ 튜닝 섹션]
# ==============================================================================

# 1. [공통] 위치 보정 (최종 오프셋)
GLOBAL_OFFSET_X = 0.0     
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0     

# 2. [개별] 아이템별 설정
ITEM_OFFSETS = {
    # [버거]
    "burger1": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "burger2": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "burger3": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    
    # [음료/사이드]
    "coke":    {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "cider":   {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "fries":   {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "nugget":  {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
}

# 3. [보정] 특정 아이템 대상 보정 설정
TARGET_CORRECTION_ITEMS = ['cider', 'coke', 'fries', 'nugget'] 

# A. X축 스케일링 (관측 위치 기준 상대 거리 확장)
# 관측 위치로부터 100mm 떨어지면 -> 105mm 이동 (1.05배)
SCALE_FACTOR_X = 1.05 

# B. Z축 기울기 보정 (단, +X 방향일 때만 적용)
# +X 방향으로 갈수록 바닥이 낮아짐(로봇이 들림) -> Z를 더 내려야 함
TILT_FACTOR_X = 0.15 

# 4. 접근 높이
APPROACH_HEIGHT = 100.0

# 5. 관측 자세 (Joint)
J_LOOK_POS = [-45.0, 20.0, 30.0, 0.0, 130.0, 135.0]

# 6. 임시 배치 위치
TEMP_PLACE_POS = [300.0, 10.0, 200.0, 0.0, 180.0, 0.0]
SAFE_Z_FLOOR_LIMIT = -15.0 

# 7. 안전 바닥 높이
SAFE_Z_FLOOR_LIMIT = -15.0 
# ==============================================================================

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

node_ = None          
dsr_control_node_ = None 
manager = None
vision_cli = None 
gripper = None
gripper_manager = None
T_GRIPPER_TO_CAM = None

def load_calibration():
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    if os.path.exists(npy_path):
        T_GRIPPER_TO_CAM = np.load(npy_path)
    else:
        sys.exit(1)

try:
    from ff_robot.onrobot import RG
except ImportError:
    class RG:
        def __init__(self, *args): pass
        def open_gripper(self): print("   👐 [Virtual] Open")
        def close_gripper(self, force=None): print("   ✊ Close")

def get_robot_pose_matrix(posx_list):
    x, y, z, rx, ry, rz = posx_list
    rot = R.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T

def transform_camera_to_base(cam_xyz):
    from DSR_ROBOT2 import get_current_posx
    try:
        curr_posx = get_current_posx()
        if curr_posx is None: return None
        if isinstance(curr_posx, tuple): curr_posx = curr_posx[0]
        
        T_base_gripper = get_robot_pose_matrix(curr_posx)
        T_base_cam = T_base_gripper @ T_GRIPPER_TO_CAM
        
        p_cam = np.array([cam_xyz[0]*1000, cam_xyz[1]*1000, cam_xyz[2]*1000, 1.0])
        p_base = T_base_cam @ p_cam
        return p_base[:3]
    except Exception as e:
        return None

def wait_for_motion():
    from DSR_ROBOT2 import check_motion
    time.sleep(0.1)
    while check_motion() != 0:
        time.sleep(0.05) 
        if not rclpy.ok(): return False
    return True

def safe_movel(pos, desc="이동"):
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    try:
        time.sleep(0.05) 
        movel(pos, vel=[50.0, 50.0], acc=[50.0, 50.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05) 
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

# ==============================================================================
# [Action] Pick 동작 (Relative Scaling & Directional Tilt)
# ==============================================================================
def safe_move_and_pick(base_pos, item_name, ref_x_pos):
    """
    ref_x_pos: 관측 위치(Look Pose)에서의 로봇 Base X좌표
    """
    global gripper_manager, node_
    
    offset_info = ITEM_OFFSETS.get(item_name, {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None})
    
    raw_x = base_pos[0]
    raw_y = base_pos[1]
    raw_z = base_pos[2]

    # ==========================================================
    # [보정 로직] 1. 관측위치 기준 X 스케일링 / 2. +X 방향만 Z 보정
    # ==========================================================
    if item_name in TARGET_CORRECTION_ITEMS:
        # 1. 관측 위치로부터의 거리(Delta)
        delta_x = raw_x - ref_x_pos
        
        # 2. X축 스케일링 (방향 상관없이 1.05배 확장)
        scaled_delta_x = delta_x * SCALE_FACTOR_X
        
        # 보정된 절대 X좌표
        target_x = ref_x_pos + scaled_delta_x + GLOBAL_OFFSET_X
        
        # 3. Z축 기울기 보정 (조건: Delta > 0, 즉 +X 방향일 때만)
        tilt_correction_z = 0.0
        
        if delta_x > 0:
            # +X 방향: 멀리 갈수록 Z를 깎음 (더 내려감)
            tilt_correction_z = scaled_delta_x * TILT_FACTOR_X
            node_.get_logger().info(
                f"   📉 [Tilt 적용] +X 방향 감지! Z 보정: -{tilt_correction_z:.1f}mm"
            )
        else:
            # -X 방향: 보정 없음
            node_.get_logger().info(
                f"   ➡️ [Tilt 패스] -X 방향임. 보정 없음."
            )

        node_.get_logger().info(
            f"   🔧 [좌표 보정] Delta({delta_x:.1f}->{scaled_delta_x:.1f}) / Final Z_Adj(-{tilt_correction_z:.1f})"
        )
        
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"] - tilt_correction_z

    else:
        # 보정 대상 아님 (버거 등)
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]
    # ==========================================================

    # 회전값
    target_rx = offset_info.get("rx", 0.0)
    target_ry = offset_info.get("ry", 180.0)
    target_rz = base_pos[5] if offset_info.get("rz") is None else offset_info["rz"]

    # 안전장치
    if target_z < SAFE_Z_FLOOR_LIMIT:
        node_.get_logger().warn(f"⚠️ Z Safe Limit: {target_z:.1f} -> {SAFE_Z_FLOOR_LIMIT}")
        target_z = SAFE_Z_FLOOR_LIMIT

    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    
    node_.get_logger().info(f"   🎯 Pick Pose: {pick_pose}")

    if not safe_movel(approach_pose, "접근"): return False
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    if not safe_movel(pick_pose, "하강"): return False
    if gripper_manager: gripper_manager.execute_grip(item_name)
    else:
        if gripper: gripper.close_gripper()
        time.sleep(0.5)
    if not safe_movel(approach_pose, "상승"): return False
    return True

# ==============================================================================
# [Task Thread] 
# ==============================================================================
def call_vision_service(target_name):
    global node_, vision_cli
    if vision_cli is None or not vision_cli.service_is_ready(): return None
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 60.0: return None
        time.sleep(0.1) 
    
    try:
        res = future.result()
        if res.found:
            return [res.position.x, res.position.y, res.position.z, res.rx, res.ry, res.rz]
    except: pass
    return None

def perform_robot_task():
    global node_, manager, gripper, gripper_manager
    try: from DSR_ROBOT2 import movej, get_current_posx
    except: return

    try: gripper = RG("rg2", "192.168.1.1", "502") 
    except: gripper = RG()
    gripper_manager = GripperManager(gripper)

    # 1. 초기 이동
    try:
        movej(J_LOOK_POS, vel=50.0, acc=50.0)
        wait_for_motion()
    except: pass
    
    node_.get_logger().info("[Task] Robot Ready.")

    while rclpy.ok():
        try:
            needed_items = manager.get_all_needed_items()
            if not needed_items:
                time.sleep(1.0) 
                continue

            target_item = needed_items[0]
            node_.get_logger().info(f"🔎 [Search] '{target_item}'")

            # 2. 관측 위치로 이동
            try:
                movej(J_LOOK_POS, vel=50.0, acc=50.0)
                wait_for_motion()
                time.sleep(1.0)
            except: pass

            # 3. [중요] 관측 위치에서의 로봇 좌표(Reference X) 저장
            try:
                curr_pos = get_current_posx()
                if isinstance(curr_pos, tuple): curr_pos = curr_pos[0]
                ref_x_pos = curr_pos[0] 
                # node_.get_logger().info(f"   📍 기준 X좌표: {ref_x_pos:.1f}")
            except:
                ref_x_pos = 0.0 

            if gripper_manager: gripper_manager.prepare_grip(target_item)

            cam_pose = call_vision_service(target_item)
            if cam_pose is None:
                time.sleep(1.0)
                continue

            base_xyz = transform_camera_to_base(cam_pose[:3])
            if base_xyz is None: continue
            
            base_pose_full = [base_xyz[0], base_xyz[1], base_xyz[2], 0, 0, cam_pose[5]]
            
            # 4. ref_x_pos를 인자로 전달
            if safe_move_and_pick(base_pose_full, target_item, ref_x_pos):
                node_.get_logger().info("✅ Pick 성공!")
                if safe_movel(TEMP_PLACE_POS, "Place"):
                    if gripper_manager: gripper_manager.release(target_item)
                    manager.clear_slot(0) 
            else:
                node_.get_logger().error("❌ Pick Failed")

            time.sleep(0.5)
        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

def handle_order_request(request, response):
    global manager, node_
    node_.get_logger().info(f"⚡ [Service] 주문: {request.item_names}, {list(request.item_quantities)}")
    success, msg = manager.check_and_deduct_stock(request.item_names, request.item_quantities)
    if success:
        manager.add_order_to_slot(f"ORD-{int(time.time())}", request.item_names, request.item_quantities)
        response.assigned_order_id = f"ORD-{int(time.time())}"
        response.message = "접수 완료"
    else:
        response.assigned_order_id = ""
        response.message = msg
    return response

def main(args=None):
    global node_, dsr_control_node_, manager, vision_cli, T_GRIPPER_TO_CAM
    rclpy.init(args=args)
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