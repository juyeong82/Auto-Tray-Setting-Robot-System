#!/usr/bin/env python3
# item_placement_controller.py
# 아이템 검출 → 집기 → 트레이 배치 전담 노드

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

import DR_init
from ff_robot_interfaces.srv import DetectObject, PlaceItem
from ff_robot.gripper import GripperManager
from ff_robot.order_logic import SlotManager

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0

TRAY_CENTER_OFFSET_X = 80.0
TRAY_0_EDGE_POS = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
TRAY_1_EDGE_POS = [435.0, 20.0, 25.0, 43.35, -180.0, -134.82]

TRAY_FLOOR_Z = 25.0
SAFE_Z_FLOOR_LIMIT = 10.0

APPROACH_HEIGHT = 100.0
EXTRA_LIFT_HEIGHT = 50.0
DROP_SAFETY_MARGIN = 30.0

J_ITEM_OBSERVE = [-34.0, 33.0, 15.0, 0.0, 132.0, 144.0]

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
vision_cli = None
gripper = None
gripper_manager = None
T_GRIPPER_TO_CAM = None
slot_manager = None  # 배치 패턴 계산용

def load_calibration():
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    if os.path.exists(npy_path):
        T_GRIPPER_TO_CAM = np.load(npy_path)
        node_.get_logger().info(f"✅ Calibration loaded: {npy_path}")
    else:
        node_.get_logger().error(f"❌ Calibration file not found: {npy_path}")
        sys.exit(1)

def get_tray_center_pose(slot_id):
    edge_pos = TRAY_0_EDGE_POS if slot_id == 0 else TRAY_1_EDGE_POS
    center_pos = list(edge_pos)
    center_pos[0] += TRAY_CENTER_OFFSET_X
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

def call_vision_service(target_name):
    if vision_cli is None or not vision_cli.service_is_ready():
        return None
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
            return {
                "position": [res.position.x, res.position.y, res.position.z],
                "rotation": [res.rx, res.ry, res.rz],
                "confidence": res.confidence
            }
    except: pass
    return None

# ==============================================================================
# SERVICE HANDLER: /place_item
# ==============================================================================
def handle_place_item(request, response):
    from DSR_ROBOT2 import get_current_posx
    
    item_name = request.item_name
    target_slot_id = request.target_slot_id
    item_index = request.item_index
    total_items = request.total_items
    
    node_.get_logger().info(f"🔧 [PlaceItem] 요청: '{item_name}' → Slot {target_slot_id} (#{item_index}/{total_items})")
    
    # 1. 관찰 자세로 이동
    if not safe_movej(J_ITEM_OBSERVE):
        response.success = False
        response.message = "Failed to move to observation pose"
        return response
    
    time.sleep(1.0)
    
    # 2. 레퍼런스 좌표 획득
    try:
        curr_pos = get_current_posx()
        if isinstance(curr_pos, tuple): curr_pos = curr_pos[0]
        ref_x_pos, ref_y_pos = curr_pos[0], curr_pos[1]
        safe_travel_z = max(curr_pos[2], 150.0)
    except:
        ref_x_pos, ref_y_pos, safe_travel_z = 0.0, 0.0, 200.0
    
    # 3. 그리퍼 준비
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    
    # 4. 비전 호출
    item_data = call_vision_service(item_name)
    if not item_data:
        response.success = False
        response.message = f"Vision failed to detect '{item_name}'"
        return response
    
    # 5. 좌표 변환
    base_xyz = transform_camera_to_base(item_data["position"])
    if base_xyz is None:
        response.success = False
        response.message = "Coordinate transformation failed"
        return response
    
    rz_obb = item_data["rotation"][2]
    full_pose = [base_xyz[0], base_xyz[1], base_xyz[2], 0, 0, rz_obb]
    
    # 6. 아이템 집기
    if not safe_move_and_pick_item(full_pose, item_name, ref_x_pos, ref_y_pos, rz_obb):
        response.success = False
        response.message = "Failed to pick item"
        return response
    
    # 7. 트레이 배치 좌표 계산
    center_tray_pos = get_tray_center_pose(target_slot_id)
    
    # SlotManager의 배치 패턴 활용
    grid_pos = slot_manager.get_tray_place_pose(center_tray_pos, item_index, total_items)
    
    approach_place = list(grid_pos)
    approach_place[2] = safe_travel_z
    
    if not safe_movel(approach_place, "배치 상공 접근"):
        response.success = False
        response.message = "Failed to approach placement position"
        return response
    
    # 8. 배치 하강
    item_h = ITEM_PLACE_Z_OFFSET.get(item_name, 30.0)
    final_z = TRAY_FLOOR_Z + item_h + DROP_SAFETY_MARGIN
    drop_pose = list(approach_place)
    drop_pose[2] = final_z
    
    if not safe_movel(drop_pose, "배치 하강"):
        response.success = False
        response.message = "Failed to descend for placement"
        return response
    
    # 9. 놓기
    if gripper_manager: gripper_manager.release(item_name)
    else:
        if gripper: gripper.open_gripper(); time.sleep(0.5)
    
    # 10. 성공 응답
    response.success = True
    response.message = "Item placed successfully"
    response.actual_x = drop_pose[0]
    response.actual_y = drop_pose[1]
    response.actual_z = drop_pose[2]
    
    node_.get_logger().info(f"   ✅ '{item_name}' 배치 완료: ({drop_pose[0]:.1f}, {drop_pose[1]:.1f}, {drop_pose[2]:.1f})")
    
    return response

def main(args=None):
    global node_, dsr_control_node_, vision_cli, gripper, gripper_manager, slot_manager
    
    rclpy.init(args=args)
    
    node_ = rclpy.create_node("item_placement_controller", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_item_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_
    
    load_calibration()
    
    # SlotManager 초기화 (배치 패턴 계산용)
    slot_manager = SlotManager()
    
    node_.get_logger().info("=========================================")
    node_.get_logger().info("🔧 Item Placement Controller Node")
    node_.get_logger().info("=========================================")
    
    # 그리퍼 초기화
    try:
        gripper = RG("rg2", "192.168.1.1", "502")
        node_.get_logger().info("✅ Real Gripper Initialized")
    except:
        gripper = RG()
        node_.get_logger().info("⚠️ Virtual Gripper Initialized")
    
    gripper_manager = GripperManager(gripper)
    
    # 서비스 설정
    cb_group = ReentrantCallbackGroup()
    node_.create_service(PlaceItem, '/place_item', handle_place_item, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)
    
    # 초기 자세 이동
    safe_movej(J_ITEM_OBSERVE)
    node_.get_logger().info("📍 Ready at J_ITEM_OBSERVE position")
    node_.get_logger().info("🎯 Service Ready: /place_item")
    
    executor = MultiThreadedExecutor()
    executor.add_node(node_)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()