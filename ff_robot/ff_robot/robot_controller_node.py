#!/usr/bin/env python3
# robot_controller_node_gem.py (Final Fix: Missing Function Added)

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
from ff_robot.tray_manager import TrayManager 

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
GLOBAL_OFFSET_X = 0.0     
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0     

# [1] 트레이 보충용 (옆구리 -X 잡기)
TRAY_PICK_X_OFFSET = 50.0   
TRAY_PICK_Z_OFFSET = 0.0    # 오프셋 제거
TRAY_PICK_HEIGHT_Z = 20.0   # 절대 높이 사용

# [2] 서빙용 (아래쪽 -Y 잡기)
SERVING_PICK_OFFSET_Y = 120.0 
SERVING_PICK_Z_OFFSET = -12.0
SERVING_GRIP_RZ = 90.0        

# [Serving] 서빙 시 미는 거리 (mm)
SERVING_PUSH_DISTANCE = 150.0 

# [트레이 좌표 - 기준점]
TRAY_0_POS = [205.0, 20.0, 20.0, 43.35, -180.0, -134.82]
TRAY_1_POS = [435.0, 20.0, 20.0, 43.35, -180.0, -134.82]

# [트레이 배치 중심 및 서빙 잡기 위치]
TRAY_CENTER_OFFSET_X = 80.0 
TRAY_0_EDGE_POS = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
TRAY_1_EDGE_POS = [435.0, 20.0, 25.0, 43.35, -180.0, -134.82]

TRAY_FLOOR_Z = 25.0  
SAFE_Z_FLOOR_LIMIT = 10.0 

# [Safe Move]
APPROACH_HEIGHT = 100.0      
EXTRA_LIFT_HEIGHT = 50.0     
DROP_SAFETY_MARGIN = 30.0    

J_TRAY_OBSERVE = [0.0, 30.0, 25.0, 0.0, 110.0, 0.0]  
J_ITEM_OBSERVE = [-34.0, 33.0, 15.0, 0.0, 132.0, 144.0] 
J_TABLE_CHECK = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0] 

ITEM_OFFSETS = {
    "burger1": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "burger2": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "burger3": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "coke":    {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "cider":   {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "fries":   {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "nugget":  {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
}

ITEM_PLACE_Z_OFFSET = {
    "burger1": 30.0, "burger2": 30.0, "burger3": 30.0, 
    "fries":   65.0, "nugget":  65.0,
    "coke":    125.0, "cider":   125.0
}

FIXED_PICK_Z = {
    "coke": 84.5, "cider": 84.5
}

SCALE_X_TARGETS = ['cider', 'coke', 'fries', 'nugget']
SCALE_Y_TARGETS = ['burger1', 'burger2', 'burger3']

SCALE_FACTOR_X = 1.05 
SCALE_FACTOR_Y = 1.05
TILT_FACTOR_X = 0.10

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

node_ = None          
dsr_control_node_ = None 
manager = None      
tray_manager = None 
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

# ==============================================================================
# [Helper] 트레이 진짜 중심 좌표 계산기 (누락된 함수 복구)
# ==============================================================================
def get_tray_center_pose(slot_id):
    # 1. 가장자리(Edge) 좌표 가져오기
    edge_pos = TRAY_0_EDGE_POS if slot_id == 0 else TRAY_1_EDGE_POS
    
    # 2. X축으로 반폭만큼 이동 (중심 찾기)
    center_pos = list(edge_pos)
    center_pos[0] += TRAY_CENTER_OFFSET_X 
    
    # 3. 회전값 정렬 (음식 놓기 좋은 0.0도)
    center_pos[3] = 0.0
    center_pos[4] = 180.0
    center_pos[5] = 0.0
    
    return center_pos

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

def safe_movej(joints):
    from DSR_ROBOT2 import movej
    try:
        time.sleep(0.05)
        movej(joints, vel=50.0, acc=50.0)
        wait_for_motion()
        time.sleep(0.05)
        return True
    except: return False

def safe_movel(pos, desc="이동"):
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    try:
        time.sleep(0.05) 
        movel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05) 
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ [{desc}] 오류: {e}")
        return False

# ==============================================================================
# [Action] 물품 집기
# ==============================================================================
def safe_move_and_pick_item(base_pos, item_name, ref_x_pos, ref_y_pos, rot_rz):
    offset_info = ITEM_OFFSETS.get(item_name, {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None})
    raw_x, raw_y, raw_z = base_pos[0], base_pos[1], base_pos[2]

    if item_name in SCALE_X_TARGETS:
        delta_x = raw_x - ref_x_pos
        target_x = ref_x_pos + (delta_x * SCALE_FACTOR_X) + GLOBAL_OFFSET_X
        tilt_z = (delta_x * TILT_FACTOR_X) if delta_x > 0 else 0.0
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"] - tilt_z
    elif item_name in SCALE_Y_TARGETS:
        delta_y = raw_y - ref_y_pos
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = ref_y_pos + (delta_y * SCALE_FACTOR_Y) + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]
    else:
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]

    if item_name in FIXED_PICK_Z:
        target_z = FIXED_PICK_Z[item_name]

    target_rx = offset_info.get("rx", 0.0)
    target_ry = offset_info.get("ry", 180.0)
    target_rz = rot_rz if offset_info.get("rz") is None else offset_info["rz"]

    if target_z < SAFE_Z_FLOOR_LIMIT: target_z = SAFE_Z_FLOOR_LIMIT

    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    lift_pose = [target_x, target_y, target_z + APPROACH_HEIGHT + EXTRA_LIFT_HEIGHT, target_rx, target_ry, target_rz]
    
    if not safe_movel(approach_pose, "아이템 접근"): return False
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    if not safe_movel(pick_pose, "아이템 하강"): return False
    if gripper_manager: gripper_manager.execute_grip(item_name)
    else:
        if gripper: gripper.close_gripper(); time.sleep(0.5)
    if not safe_movel(lift_pose, "아이템 상승(High)"): return False
    return True

# ==============================================================================
# [Action] 트레이 집기 (옆구리 잡기 -X -> EDGE_POS로 배치)
# ==============================================================================
def pick_and_place_tray(detected_data, slot_id):
    global tray_manager, gripper
    
    cam_pos = detected_data["position"]
    base_pos = transform_camera_to_base(cam_pos) 
    rotation_rz = detected_data["rotation"][2] 
    
    if base_pos is None: return False

    # [1] 트레이 집기 (옆구리)
    grip_pose = tray_manager.calculate_tray_grip_point(base_pos, rotation_rz)
    grip_pose[0] += TRAY_PICK_X_OFFSET  
    
    # Z축 강제 고정
    FORCED_Z = TRAY_PICK_HEIGHT_Z + TRAY_PICK_Z_OFFSET
    grip_pose[2] = FORCED_Z
    
    node_.get_logger().info(f"   🍱 Tray Side Grip Pose: {grip_pose}")
    
    approach_pose = grip_pose[:]
    approach_pose[2] += APPROACH_HEIGHT
    
    if gripper: gripper.open_gripper()
    if not safe_movel(approach_pose, "트레이 접근(상공)"): return False
    if not safe_movel(grip_pose, "트레이 잡기 위치 하강"): return False
    if gripper: gripper.close_gripper(); time.sleep(1.0)
    
    lift_pose = grip_pose[:]
    lift_pose[2] = 250.0 
    if not safe_movel(lift_pose, "트레이 높게 들기"): return False

    # [3] 배치 위치: EDGE 좌표 사용 (여기로 끌고 옴)
    target_pos = TRAY_0_EDGE_POS if slot_id == 0 else TRAY_1_EDGE_POS
    
    place_pose = list(target_pos)
    # 잡은 회전각 그대로 유지하며 이동
    place_pose[3] = grip_pose[3]
    place_pose[4] = grip_pose[4]
    place_pose[5] = grip_pose[5]
    
    air_pose = place_pose[:]
    air_pose[2] = 250.0
    
    if not safe_movel(air_pose, "트레이 공중 이동"): return False
    if not safe_movel(place_pose, "트레이 배치 하강"): return False
    
    if gripper: gripper.open_gripper(); time.sleep(0.5)
    
    depart_pose = place_pose[:]
    depart_pose[2] += APPROACH_HEIGHT
    if not safe_movel(depart_pose, "작업 완료 상승"): return False
    
    return True

# ==============================================================================
# [Action] 서빙 (아래 잡고 밀기 -Y)
# ==============================================================================
def serve_tray(slot_id):
    global tray_manager, gripper
    
    # [수정] 진짜 중심 좌표를 가져와서 사용
    center_pos = get_tray_center_pose(slot_id)
    
    # [아래쪽 잡기]
    grip_pose = list(center_pos)
    grip_pose[1] -= SERVING_PICK_OFFSET_Y # 아래쪽(-Y)으로 이동
    grip_pose[2] = TRAY_FLOOR_Z + SERVING_PICK_Z_OFFSET 
    grip_pose[5] = SERVING_GRIP_RZ # 가로 회전
    
    approach = list(grip_pose)
    approach[2] += APPROACH_HEIGHT
    
    node_.get_logger().info(f"   🍽️ 서빙 시작 (Slot {slot_id}) - 좌표: {grip_pose[:3]}")
    
    if gripper: gripper.open_gripper()
    if not safe_movel(approach, "서빙 준비 접근"): return False
    if not safe_movel(grip_pose, "서빙 그립 하강"): return False
    if gripper: gripper.close_gripper(); time.sleep(1.0)
    
    # 2. 밀기
    push_pose = list(grip_pose)
    push_pose[1] += SERVING_PUSH_DISTANCE # +Y 방향
    
    node_.get_logger().info(f"   🚀 서빙 밀기 (+Y {SERVING_PUSH_DISTANCE}mm)")
    
    if not safe_movel(push_pose, "서빙 밀기 동작"): return False
    
    # 3. 놓기 및 복귀
    if gripper: gripper.open_gripper(); time.sleep(0.5)
    
    depart = list(push_pose)
    depart[2] += APPROACH_HEIGHT
    
    if not safe_movel(depart, "서빙 완료 후 상승"): return False
    
    return True

def call_vision_service(target_name):
    if vision_cli is None or not vision_cli.service_is_ready(): return None
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 5.0: return None
        time.sleep(0.1) 
    try:
        res = future.result()
        if res.found:
            return {"position": [res.position.x, res.position.y, res.position.z], "rotation": [res.rx, res.ry, res.rz], "confidence": res.confidence}
    except: pass
    return None

# ==============================================================================
# MAIN TASK LOOP
# ==============================================================================
def perform_robot_task():
    global manager, gripper_manager, tray_manager, gripper
    try: from DSR_ROBOT2 import movej, get_current_posx
    except: return

    try: 
        gripper = RG("rg2", "192.168.1.1", "502") 
        node_.get_logger().info("✅ Real Gripper Initialized")
    except: 
        gripper = RG()
        node_.get_logger().info("⚠️ Virtual Gripper Initialized")

    gripper_manager = GripperManager(gripper)

    safe_movej(J_TRAY_OBSERVE)
    node_.get_logger().info("[Task] Robot System Ready.")

    while rclpy.ok():
        try:
            # Phase 1: 트레이 준비
            target_tray_slot = tray_manager.get_empty_slot()
            if target_tray_slot is not None:
                node_.get_logger().info(f"🔎 [Phase 1] 선제적 트레이 준비 (Slot {target_tray_slot})")
                safe_movej(J_TRAY_OBSERVE)
                time.sleep(1.0)
                tray_data = call_vision_service("tray")
                if tray_data:
                    if pick_and_place_tray(tray_data, target_tray_slot):
                        if manager.pending_queue and manager.active_slots[target_tray_slot] is None:
                            manager.clear_slot(target_tray_slot)
                        tray_manager.update_tray_status(target_tray_slot, "working")
                        continue

            # Phase 2: 물품 배치
            needed_items = manager.get_all_needed_items()
            if needed_items:
                target_item = needed_items[0]
                
                valid_target = False
                for sid, data in manager.active_slots.items():
                    if data and target_item in data['needed'] and \
                       data['placed_total'].count(target_item) < data['needed'].count(target_item):
                        if tray_manager.tray_states[sid]['status'] == 'working':
                            valid_target = True
                            break
                
                if valid_target:
                    node_.get_logger().info(f"🔎 [Phase 2] 물품 탐색: '{target_item}'")
                    safe_movej(J_ITEM_OBSERVE)
                    time.sleep(1.0)
                    
                    try:
                        curr_pos = get_current_posx()
                        if isinstance(curr_pos, tuple): curr_pos = curr_pos[0]
                        ref_x_pos, ref_y_pos = curr_pos[0], curr_pos[1]
                        safe_travel_z = max(curr_pos[2], 150.0)
                    except: ref_x_pos, ref_y_pos, safe_travel_z = 0.0, 0.0, 200.0

                    if gripper_manager: gripper_manager.prepare_grip(target_item)
                    item_data = call_vision_service(target_item)
                    
                    if item_data:
                        base_xyz = transform_camera_to_base(item_data["position"])
                        if base_xyz is not None:
                            rz_obb = item_data["rotation"][2]
                            full_pose = [base_xyz[0], base_xyz[1], base_xyz[2], 0, 0, rz_obb]
                            
                            if safe_move_and_pick_item(full_pose, target_item, ref_x_pos, ref_y_pos, rz_obb):
                                target_slot = None
                                for sid, data in manager.active_slots.items():
                                    if data and target_item in data['needed'] and \
                                       data['placed_total'].count(target_item) < data['needed'].count(target_item):
                                        target_slot = sid
                                        break
                                
                                if target_slot is not None:
                                    center_tray_pos = get_tray_center_pose(target_slot)
                                    slot_data = manager.active_slots[target_slot]
                                    current_idx = len(slot_data['placed_total'])
                                    total_count = len(slot_data['needed'])
                                    
                                    grid_pos = manager.get_tray_place_pose(center_tray_pos, current_idx, total_count)
                                    
                                    approach_place = list(grid_pos)
                                    approach_place[2] = safe_travel_z 
                                    
                                    if safe_movel(approach_place, "배치 상공 접근"):
                                        item_h = ITEM_PLACE_Z_OFFSET.get(target_item, 30.0)
                                        final_z = TRAY_FLOOR_Z + item_h + DROP_SAFETY_MARGIN
                                        drop_pose = list(approach_place)
                                        drop_pose[2] = final_z
                                        
                                        if safe_movel(drop_pose, "배치 하강"):
                                            if gripper_manager: gripper_manager.release(target_item)
                                            is_done = manager.mark_item_done(target_slot, target_item)
                                            if is_done:
                                                tray_manager.update_tray_status(target_slot, "ready")
                else:
                    time.sleep(0.5)

            # Phase 3: 서빙
            ready_slot = tray_manager.get_ready_slot()
            if ready_slot is not None:
                node_.get_logger().info(f"🍽️ [Phase 3] 서빙 시작 (Slot {ready_slot})")
                if serve_tray(ready_slot):
                    manager.clear_slot(ready_slot)
                    tray_manager.update_tray_status(ready_slot, "empty")
            
            time.sleep(0.5)
            
        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

def handle_order_request(request, response):
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
    global node_, dsr_control_node_, manager, tray_manager, vision_cli, T_GRIPPER_TO_CAM
    rclpy.init(args=args)
    load_calibration()
    
    manager = SlotManager()
    tray_manager = TrayManager()
    
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