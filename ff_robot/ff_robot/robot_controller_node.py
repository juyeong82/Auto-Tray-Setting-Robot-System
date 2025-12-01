#!/usr/bin/env python3
# robot_controller_node_gem.py (X/Y Symmetric Scaling & Tilt)

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

# --------------------------
# 🎛️ CONFIGURATION (튜닝 섹션)
# --------------------------
GLOBAL_OFFSET_X = 0.0     
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0     

ITEM_OFFSETS = {
    "burger1": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "burger2": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "burger3": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "coke":    {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "cider":   {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "fries":   {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "nugget":  {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
}

# 1. 보정 대상 그룹
SCALE_X_TARGETS = ['cider', 'coke', 'fries', 'nugget']   # X 스케일링 대상
SCALE_Y_TARGETS = ['burger1', 'burger2', 'burger3']      # Y 스케일링 대상 (버거)

# 2. X축 스케일링 (거리 비례 확장)
# +X, -X 방향 모두 1.05배 적용 (100mm 이동 시 105mm 이동)
SCALE_FACTOR_X = 1.05 

# 3. Y축 스케일링 (거리 비례 확장 - 버거 전용)
# 0.75cm 더 간다는 요청 -> 약 1.05배 설정 (거리 비례)
SCALE_FACTOR_Y = 1.05

# 4. Z축 기울기 보정 (+X 방향일 때만 적용)
TILT_FACTOR_X = 0.10

APPROACH_HEIGHT = 100.0
J_LOOK_POS = [-45.0, 20.0, 30.0, 0.0, 130.0, 135.0]
TEMP_PLACE_POS = [300.0, 10.0, 200.0, 0.0, 180.0, 0.0]
SAFE_Z_FLOOR_LIMIT = -15.0 

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# Global Handles
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
# [핵심 로직] 보정 적용된 Pick 계산
# ==============================================================================
def safe_move_and_pick(base_pos, item_name, ref_x_pos, ref_y_pos):
    """
    Apply Scale & Tilt Corrections based on Reference Pose (Look Pose)
    """
    offset_info = ITEM_OFFSETS.get(item_name, {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None})
    
    raw_x, raw_y, raw_z = base_pos[0], base_pos[1], base_pos[2]

    # --- 1. X축 보정 (음료/사이드 등) ---
    if item_name in SCALE_X_TARGETS:
        delta_x = raw_x - ref_x_pos
        # [수정됨] +X, -X 방향 모두 1.05배 적용
        scaled_delta_x = delta_x * SCALE_FACTOR_X
        target_x = ref_x_pos + scaled_delta_x + GLOBAL_OFFSET_X
        
        # Z Tilt 보정 (+X 방향일 때만)
        tilt_correction_z = 0.0
        if delta_x > 0: 
            tilt_correction_z = scaled_delta_x * TILT_FACTOR_X
            node_.get_logger().info(f"   📉 [Tilt] +X detected. Z-Adjust: -{tilt_correction_z:.1f}mm")
        
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"] - tilt_correction_z

    # --- 2. Y축 보정 (버거) ---
    elif item_name in SCALE_Y_TARGETS:
        # [추가됨] Y축 거리 비례 보정
        delta_y = raw_y - ref_y_pos
        scaled_delta_y = delta_y * SCALE_FACTOR_Y
        
        node_.get_logger().info(f"   🍔 [Burger Y-Scale] Delta({delta_y:.1f}) -> Scaled({scaled_delta_y:.1f})")
        
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = ref_y_pos + scaled_delta_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]
    
    # --- 3. 보정 없음 ---
    else:
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]

    # --- Rotation ---
    target_rx = offset_info.get("rx", 0.0)
    target_ry = offset_info.get("ry", 180.0)
    target_rz = base_pos[5] if offset_info.get("rz") is None else offset_info["rz"]

    if target_z < SAFE_Z_FLOOR_LIMIT: target_z = SAFE_Z_FLOOR_LIMIT

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

def call_vision_service(target_name):
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
    global manager, gripper_manager
    try: from DSR_ROBOT2 import movej, get_current_posx
    except: return

    try: gripper = RG("rg2", "192.168.1.1", "502") 
    except: gripper = RG()
    gripper_manager = GripperManager(gripper)

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

            try:
                movej(J_LOOK_POS, vel=50.0, acc=50.0)
                wait_for_motion()
                time.sleep(1.0)
            except: pass

            # [수정됨] 기준점 X, Y 모두 저장
            try:
                curr_pos = get_current_posx()
                if isinstance(curr_pos, tuple): curr_pos = curr_pos[0]
                ref_x_pos = curr_pos[0] 
                ref_y_pos = curr_pos[1]
            except: 
                ref_x_pos = 0.0 
                ref_y_pos = 0.0

            if gripper_manager: gripper_manager.prepare_grip(target_item)

            cam_pose = call_vision_service(target_item)
            if cam_pose is None:
                time.sleep(1.0)
                continue

            base_xyz = transform_camera_to_base(cam_pose[:3])
            if base_xyz is None: continue
            
            base_pose_full = [base_xyz[0], base_xyz[1], base_xyz[2], 0, 0, cam_pose[5]]
            
            # ref_y_pos 추가 전달
            if safe_move_and_pick(base_pose_full, target_item, ref_x_pos, ref_y_pos):
                node_.get_logger().info("✅ Pick Success")
                if safe_movel(TEMP_PLACE_POS, "Place"):
                    if gripper_manager: gripper_manager.release(target_item)
                    
                    is_order_finished = manager.mark_item_done(0, target_item)
                    
                    if is_order_finished:
                        manager.clear_slot(0) 
                        node_.get_logger().info("🎉 Order Complete!")
                    else:
                        node_.get_logger().info(f"✨ '{target_item}' Done. Next item...")
            else:
                node_.get_logger().error("❌ Pick Failed")

            time.sleep(0.5)
        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

def handle_order_request(request, response):
    node_.get_logger().info(f"⚡ [Service] Order: {request.item_names}")
    success, msg = manager.check_and_deduct_stock(request.item_names, request.item_quantities)
    if success:
        manager.add_order_to_slot(f"ORD-{int(time.time())}", request.item_names, request.item_quantities)
        response.assigned_order_id = f"ORD-{int(time.time())}"
        response.message = "OK"
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