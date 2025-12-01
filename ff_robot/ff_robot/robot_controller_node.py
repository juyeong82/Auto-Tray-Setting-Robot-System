#!/usr/bin/env python3
<<<<<<< HEAD
# robot_controller_node_gem.py (Updated Item Heights)
=======
# robot_controller_node.py (Threaded Fix + GripperManager + 2-step Pick)
# MISSION 
# 음성 매끄럽게!!!
# 키오스크 주문 리스트 수량 확인!!!
>>>>>>> 4ae35a520417ccb3ac51484cb30ba66c2cb14b2e

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
# 🎛️ CONFIGURATION
# ==============================================================================
GLOBAL_OFFSET_X = 0.0     
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0     

# [Pick 오프셋] 잡을 때 보정치
ITEM_OFFSETS = {
    "burger1": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "burger2": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "burger3": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "coke":    {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "cider":   {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "fries":   {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "nugget":  {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
}

# [Place 높이 설정] 아이템 높이 (단위: mm)
# 트레이 바닥(-24.0) + 이 높이 + 안전마진(5.0) 위치에 놓게 됨
ITEM_PLACE_Z_OFFSET = {
    "burger1": 30.0,  "burger2": 30.0,  "burger3": 30.0,  # 3cm
    "fries":   65.0,  "nugget":  65.0,                    # 6.5cm
    "coke":    125.0, "cider":   125.0                    # 12.5cm
}

SCALE_X_TARGETS = ['cider', 'coke', 'fries', 'nugget']
SCALE_Y_TARGETS = ['burger1', 'burger2', 'burger3']

SCALE_FACTOR_X = 1.05 
SCALE_FACTOR_Y = 1.05
TILT_FACTOR_X = 0.10

# [Pick 동작 관련]
APPROACH_HEIGHT = 100.0      
EXTRA_LIFT_HEIGHT = 30.0     # Pick 후 3cm 더 상승

J_LOOK_POS = [-45.0, 20.0, 30.0, 0.0, 130.0, 135.0]

# [트레이 좌표]
TRAY_0_POS = [305.0, 20.0, 0.0, 43.35, -180.0, -135.0]
TRAY_1_POS = [535.0, 20.0, 0.0, 43.35, -180.0, -135.0]

# [바닥 및 안전 상수]
TRAY_FLOOR_Z = -24.0     # 실측된 트레이 바닥 높이
DROP_SAFETY_MARGIN = 5.0 # 바닥 충돌 방지용 여유 (5mm 띄워서 놓기)

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

# Pick 동작 (상승 포함)
def safe_move_and_pick(base_pos, item_name, ref_x_pos, ref_y_pos):
    offset_info = ITEM_OFFSETS.get(item_name, {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None})
    raw_x, raw_y, raw_z = base_pos[0], base_pos[1], base_pos[2]

    # 보정 로직
    if item_name in SCALE_X_TARGETS:
        delta_x = raw_x - ref_x_pos
        scaled_delta_x = delta_x * SCALE_FACTOR_X
        target_x = ref_x_pos + scaled_delta_x + GLOBAL_OFFSET_X
        tilt_correction_z = 0.0
        if delta_x > 0: 
            tilt_correction_z = scaled_delta_x * TILT_FACTOR_X
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"] - tilt_correction_z
    elif item_name in SCALE_Y_TARGETS:
        delta_y = raw_y - ref_y_pos
        scaled_delta_y = delta_y * SCALE_FACTOR_Y
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = ref_y_pos + scaled_delta_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]
    else:
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]

    target_rx = offset_info.get("rx", 0.0)
    target_ry = offset_info.get("ry", 180.0)
    target_rz = base_pos[5] if offset_info.get("rz") is None else offset_info["rz"]

    if target_z < SAFE_Z_FLOOR_LIMIT: target_z = SAFE_Z_FLOOR_LIMIT

    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    lift_pose = [target_x, target_y, target_z + APPROACH_HEIGHT + EXTRA_LIFT_HEIGHT, target_rx, target_ry, target_rz]
    
    node_.get_logger().info(f"   🎯 Pick: {pick_pose}")

    if not safe_movel(approach_pose, "접근"): return False
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    if not safe_movel(pick_pose, "하강"): return False
    if gripper_manager: gripper_manager.execute_grip(item_name)
    else:
        if gripper: gripper.close_gripper()
        time.sleep(0.5)
        
    # 집고 나서 높게 상승 (현재 이 높이가 안전 높이가 됨)
    if not safe_movel(lift_pose, "상승 (High)"): return False
    
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

            node_.get_logger().info("🔭 관측 위치로 이동 중...")
            try:
                movej(J_LOOK_POS, vel=50.0, acc=50.0)
                wait_for_motion()
                time.sleep(0.5)
            except: pass

            target_item = None
            cam_pose = None
            
            for item in needed_items:
                node_.get_logger().info(f"🔎 [Check] '{item}' 찾는 중...")
                pose_result = call_vision_service(item)
                if pose_result is not None:
                    target_item = item
                    cam_pose = pose_result
                    node_.get_logger().info(f"✨ '{item}' 발견! 작업 시작.")
                    break 
                else:
                    node_.get_logger().info(f"   💨 '{item}' 안 보임.")
            
            if target_item is None or cam_pose is None:
                node_.get_logger().warn("💤 물체 없음. 대기 중...")
                time.sleep(1.0)
                continue
            
            if gripper_manager: gripper_manager.prepare_grip(target_item)

            try:
                curr_pos = get_current_posx()
                if isinstance(curr_pos, tuple): curr_pos = curr_pos[0]
                ref_x_pos = curr_pos[0] 
                ref_y_pos = curr_pos[1]
            except: 
                ref_x_pos = 0.0 
                ref_y_pos = 0.0

            base_xyz = transform_camera_to_base(cam_pose[:3])
            if base_xyz is None: continue
            
            base_pose_full = [base_xyz[0], base_xyz[1], base_xyz[2], 0, 0, cam_pose[5]]
            
            # [Pick 실행] (집고 나서 높은 곳에 떠있음)
            if safe_move_and_pick(base_pose_full, target_item, ref_x_pos, ref_y_pos):
                node_.get_logger().info("✅ Pick Success")
                
                # 1. 목적지 결정
                target_slot_id = manager.peek_target_slot(target_item)
                base_tray_pos = TRAY_0_POS if target_slot_id == 0 else TRAY_1_POS
                slot_data = manager.active_slots[target_slot_id]
                current_idx = len(slot_data['placed_total'])
                total_count = len(slot_data['needed'])
                
                # 2. X, Y 목적지 좌표 계산 (Z는 무시)
                dest_pose = manager.get_tray_place_pose(base_tray_pos, current_idx, total_count)
                
                # ==========================================================
                # [동적 안전 이동] Current Z 유지 -> 계산된 Z로 하강
                # ==========================================================
                
                # A. 현재 로봇 높이 확인 (집고 올라온 높이)
                try:
                    current_robot_pos = get_current_posx()
                    if isinstance(current_robot_pos, tuple): current_robot_pos = current_robot_pos[0]
                    safe_travel_z = current_robot_pos[2] # 현재 높이를 안전 높이로 사용
                except:
                    safe_travel_z = 200.0 # 실패 시 기본 안전 높이
                
                # B. 상공 수평 이동 좌표 (Z = safe_travel_z)
                approach_pose = list(dest_pose)
                approach_pose[2] = safe_travel_z 
                
                node_.get_logger().info(f"🧱 [Approach] 높이 유지 이동 (Z={safe_travel_z:.1f})")
                
                if safe_movel(approach_pose, "Place Approach"):
                    
                    # C. 하강 좌표 계산 (바닥 + 아이템 높이 + 안전마진)
                    offset_z = ITEM_PLACE_Z_OFFSET.get(target_item, 30.0) # 기본값 3cm
                    final_z = TRAY_FLOOR_Z + offset_z + DROP_SAFETY_MARGIN
                    
                    drop_pose = list(approach_pose)
                    drop_pose[2] = final_z
                    
                    node_.get_logger().info(f"⬇️ [Descend] 하강 (Z={final_z:.1f}): {target_item}")
                    
                    # D. 수직 하강
                    if safe_movel(drop_pose, "Place Drop"):
                        if gripper_manager: gripper_manager.release(target_item)
                        
                        processed_slot, is_finished = manager.process_item_and_check_complete(target_item)
                        
                        if processed_slot is not None:
                            if is_finished:
                                node_.get_logger().info(f"🎉 Slot {processed_slot} Finished!")
                                manager.clear_slot(processed_slot) 
                            else:
                                node_.get_logger().info(f"✨ Slot {processed_slot}: Placed.")
                # ==========================================================

            else:
                node_.get_logger().error("❌ Pick Failed")

            time.sleep(0.5)
        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

def handle_order_request(request, response):
<<<<<<< HEAD
    node_.get_logger().info(f"⚡ [Service] Order: {request.item_names}")
=======
    global manager, node_
    node_.get_logger().info(f"⚡ [Service] 주문: {request.item_names}, {list(request.item_quantities)}")
>>>>>>> 4ae35a520417ccb3ac51484cb30ba66c2cb14b2e
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