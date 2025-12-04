#!/usr/bin/env python3
# module1_item.py
# ------------------------------------------------------------------------------
# [Role] 아이템 검출(Vision) → 집기(Pick) → 트레이 배치(Place) 전담 노드
# [Strategy] 2-Stage Vision (Top-Down View) + Fixed Height Movement
# ------------------------------------------------------------------------------
import math
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

# ========== 기존 추가한 부분을 아래로 교체 ==========
from std_msgs.msg import Float64MultiArray
from dsr_msgs2.srv import MovePause, MoveResume
# =================================================

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================

# [Robot Base Offset] 로봇 베이스와 작업대 간의 미세 보정값 -> safe_move_and_pick_item() 내 좌표 계산 시 더해짐
GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0

# [Tray Positions] 트레이 배치 관련 좌표
# TRAY_CENTER_OFFSET_X: 트레이 모서리에서 중심까지의 거리 (X축)
# TRAY_0/1_EDGE_POS: 각 슬롯(0, 1)에 놓인 트레이를 집는 기준점(모서리 중심) 좌표
TRAY_CENTER_OFFSET_X = 80.0
TRAY_0_EDGE_POS = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
TRAY_1_EDGE_POS = [435.0, 20.0, 25.0, 43.35, -180.0, -134.82]

TRAY_FLOOR_Z = -25.0    # TRAY_FLOOR_Z: 아이템을 내려놓을 때 기준이 되는 바닥 높이 (낮을수록 더 내려감)
SAFE_Z_FLOOR_LIMIT = 10.0


EXTRA_LIFT_HEIGHT = 150.0
DROP_SAFETY_MARGIN = 0.0  # 원래 7

J_ITEM_OBSERVE = [-33.197, 21.512, 35.707, -0.118, 122.787, 144.296]

# J_ITEM_CHECKPOINT = [-45.4, 27.69, 55.06, 0.00, 97.67, 132.68] 


ITEM_OFFSETS = {
    "burger1": {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "burger2": {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "burger3": {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "coke":    {"z": 10.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "cider":   {"z": 10.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "fries":   {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "nugget":  {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
}

ITEM_HEIGHTS = {
    "burger1": 30.0, "burger2": 30.0, "burger3": 30.0,
    "fries":   65.0, "nugget":  65.0,
    "coke":    125.0, "cider":   125.0
}

FIXED_PICK_Z = {}
# {
#     "coke": 92.0, "cider": 92.0, "burger1": 50.0, "burger2": 50.0, "burger3": 50.0, 
# }

SCALE_X_TARGETS = ['cider', 'coke', 'fries', 'nugget']
SCALE_Y_TARGETS = ['burger1', 'burger2', 'burger3']

SCALE_FACTOR_X = 1.05
SCALE_FACTOR_Y = 1.05
TILT_FACTOR_X = 0.10

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# ========== 추가 ==========
# [Force Monitor Config]
FORCE_THRESHOLD = 20.0  # N
MOVING_AVG_WINDOW = 5
COOLDOWN_TIME = 1.0  # seconds
# ==========================

node_ = None
dsr_control_node_ = None
vision_cli = None
gripper = None
gripper_manager = None
T_GRIPPER_TO_CAM = None
slot_manager = None

# ========== 추가 ==========
cli_move_pause = None
cli_move_resume = None
is_paused = False
force_buffer = []
last_trigger_time = 0.0
# ==========================

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
    center_pos[3] = 56.0
    center_pos[4] = 179.87
    center_pos[5] = -126.01
    return center_pos

try:
    from ff_robot.onrobot import RG
except ImportError:
    class RG:
        def __init__(self, *args): pass
        def open_gripper(self): print("   👐 [Virtual] Open")
        def close_gripper(self, force=None): print("   ✊ [Virtual] Close")

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

# def wait_for_motion():
#     from DSR_ROBOT2 import check_motion
#     time.sleep(0.1)
#     while check_motion() != 0:
#         time.sleep(0.05)
#         if not rclpy.ok(): return False
#     return True

# ========== 기존 함수 전체 교체 ==========
def wait_for_motion():
    from DSR_ROBOT2 import check_motion
    global is_paused
    
    time.sleep(0.1)
    while check_motion() != 0:
        # Pause 상태면 대기
        while is_paused and rclpy.ok():
            time.sleep(0.1)
        
        time.sleep(0.05)
        if not rclpy.ok(): 
            return False
    return True
# ========================================

def safe_movej(joints):
    from DSR_ROBOT2 import amovej
    try:
        time.sleep(0.05)
        amovej(joints, vel=50.0, acc=50.0)
        
        # wait_for_motion 실행 결과 확인
        if not wait_for_motion(): 
            node_.get_logger().error("⚠️ wait_for_motion returned False during safe_movej")
            return False
            
        time.sleep(0.05)
        return True
    except Exception as e:
        # ⭐⭐⭐ 여기에 에러 내용을 출력하도록 수정 ⭐⭐⭐
        node_.get_logger().error(f"🔥 safe_movej Exception: {e}")
        return False

def safe_movel(pos, desc="이동"):
    from DSR_ROBOT2 import amovel, DR_BASE, DR_MV_MOD_ABS
    try:
        node_.get_logger().info(f"      → {desc} 시작: ...")
        time.sleep(0.05)
        amovel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        
        if not wait_for_motion(): 
            node_.get_logger().error(f"⚠️ wait_for_motion returned False during {desc}")
            return False
            
        time.sleep(0.05)
        node_.get_logger().info(f"      ✓ {desc} 완료")
        return True
    except Exception as e:
        # ⭐⭐⭐ 에러 내용 출력 ⭐⭐⭐
        node_.get_logger().error(f"   ❌ [{desc}] 오류 (Exception): {e}")
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

    high_z_pos = target_z + EXTRA_LIFT_HEIGHT

    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    
    approach_pose = [target_x, target_y, high_z_pos, target_rx, target_ry, target_rz]
    
    lift_pose = [target_x, target_y, high_z_pos, target_rx, target_ry, target_rz]
    
    node_.get_logger().info(f"   🛡️ 고공 접근 설정: Z={high_z_pos:.1f} (Target Z={target_z:.1f})")
    
    if not safe_movel(approach_pose, "아이템 고공 접근"): return False, None
    
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    
    if not safe_movel(pick_pose, "아이템 하강"): return False, None
    
    if gripper_manager:
        try:
            gripper_manager.execute_grip(item_name)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper execute failed (continuing): {e}")
    else:
        if gripper:
            try:
                gripper.close_gripper()
                time.sleep(0.5)
            except Exception as e:
                node_.get_logger().warn(f"⚠️ Gripper close failed (continuing): {e}")
    
    if not safe_movel(lift_pose, "아이템 상승(High)"): return False, None
    
    return True, high_z_pos

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

# ========== force_callback() 함수 전체 교체 ==========
def force_callback(msg):
    """외력 토픽 콜백 - 자동 pause/resume"""
    global is_paused, force_buffer, last_trigger_time, cli_move_pause, cli_move_resume
    
    # Float64MultiArray 구조: msg.data = [fx, fy, fz, tx, ty, tz] (보통 6축)
    if len(msg.data) < 3:
        node_.get_logger().warn("⚠️ Tool force data incomplete", throttle_duration_sec=5.0)
        return
    
    # 힘의 크기 계산 (첫 3개 요소: fx, fy, fz)
    fx, fy, fz = msg.data[0], msg.data[1], msg.data[2]
    force_magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
    
    # 이동평균
    force_buffer.append(force_magnitude)
    if len(force_buffer) > MOVING_AVG_WINDOW:
        force_buffer.pop(0)
    avg_force = np.mean(force_buffer)
    
    # 쿨다운 체크
    current_time = time.time()
    if current_time - last_trigger_time < COOLDOWN_TIME:
        return
    
    # 외력 감지 -> 상태 전환
    if avg_force > FORCE_THRESHOLD:
        last_trigger_time = current_time
        
        if not is_paused:
            # PAUSE
            node_.get_logger().warn(f"⛔ Force {avg_force:.1f}N (fx={fx:.1f}, fy={fy:.1f}, fz={fz:.1f}) -> PAUSING")
            if cli_move_pause and cli_move_pause.service_is_ready():
                req = MovePause.Request()
                cli_move_pause.call_async(req)
                is_paused = True
        else:
            # RESUME
            node_.get_logger().info(f"✅ Force {avg_force:.1f}N (fx={fx:.1f}, fy={fy:.1f}, fz={fz:.1f}) -> RESUMING")
            if cli_move_resume and cli_move_resume.service_is_ready():
                req = MoveResume.Request()
                cli_move_resume.call_async(req)
                is_paused = False
# ===================================================

def handle_place_item(request, response):
    
    item_name = request.item_name
    target_slot_id = request.target_slot_id
    item_index = request.item_index
    total_items = request.total_items
    
    node_.get_logger().info(f"🔧 [PlaceItem] 요청: '{item_name}' (2-Stage Vision)")
    

    if not safe_movej(J_ITEM_OBSERVE):
        response.success = False; response.message = "Failed to move to observation pose"
        return response
    
    time.sleep(1.0)
    

    item_data_1 = call_vision_service(item_name)
    if not item_data_1:
        response.success = False; response.message = f"1st Vision failed: '{item_name}'"
        return response
    
    base_xyz_1 = transform_camera_to_base(item_data_1["position"])
    if base_xyz_1 is None:
        response.success = False; response.message = "1st Coord transform failed"
        return response


    RE_CHECK_Z = 225.0 
    
    pre_approach_pose = [base_xyz_1[0], base_xyz_1[1], RE_CHECK_Z, 0.0, 180.0, 0.0]
    

    if not safe_movel(pre_approach_pose, "정밀 관측 위치 이동"):
        response.success = False; response.message = "Failed to move to re-check pose"
        return response
        
    time.sleep(0.5)


    item_data_2 = call_vision_service(item_name)
    
    final_x, final_y, final_z, final_rz = 0.0, 0.0, 0.0, 0.0
    
    if item_data_2:
        base_xyz_2 = transform_camera_to_base(item_data_2["position"])
        if base_xyz_2 is not None:
            final_x = base_xyz_2[0]
            final_y = base_xyz_2[1]
            final_z = base_xyz_2[2]
            final_rz = item_data_2["rotation"][2]
            node_.get_logger().info(f"   🎯 2차 보정 완료: Z값 {base_xyz_1[2]:.1f} -> {final_z:.1f}")
        else:
            final_x, final_y, final_z = base_xyz_1
            final_rz = item_data_1["rotation"][2]
    else:
        node_.get_logger().warn("   ⚠️ 2차 인식 실패 -> 1차 좌표 사용")
        final_x, final_y, final_z = base_xyz_1
        final_rz = item_data_1["rotation"][2]

    if final_z < SAFE_Z_FLOOR_LIMIT: 
        final_z = SAFE_Z_FLOOR_LIMIT


    if gripper_manager: gripper_manager.prepare_grip(item_name)
    
    pick_rz = final_rz 
    full_pose = [final_x, final_y, final_z, 0, 0, pick_rz]
    

    curr_pos = pre_approach_pose
    ref_x_pos, ref_y_pos = curr_pos[0], curr_pos[1]
    

    success, lifted_z = safe_move_and_pick_item(full_pose, item_name, ref_x_pos, ref_y_pos, pick_rz)
    
    if not success:
        response.success = False; response.message = "Failed to pick item"
        return response

    center_tray_pos = get_tray_center_pose(target_slot_id)
    center_tray_pos[1] += 10
    grid_pos = slot_manager.get_tray_place_pose(center_tray_pos, item_index, total_items)
    
    grid_pos[5] += 90.0 

    approach_place = list(grid_pos)
    approach_place[2] = lifted_z
    
    if not safe_movel(approach_place, "배치 상공 접근"):
        response.success = False; response.message = "Failed approach"
        return response
    
    item_h = ITEM_HEIGHTS.get(item_name, 30.0) 
    final_place_z = TRAY_FLOOR_Z + item_h + DROP_SAFETY_MARGIN
    drop_pose = list(approach_place)
    drop_pose[2] = final_place_z
    
    if not safe_movel(drop_pose, "배치 하강"):
        response.success = False; response.message = "Failed descend"
        return response

    if gripper_manager: gripper_manager.release(item_name)
    else: 
        if gripper: gripper.open_gripper(); time.sleep(0.5)
        
    depart_pose = list(drop_pose)
    depart_pose[2] = lifted_z
    safe_movel(depart_pose, "배치 후 상승")
    
    response.success = True
    response.message = "Success (2-Stage Vision)"
    response.actual_x = drop_pose[0]
    response.actual_y = drop_pose[1]
    response.actual_z = drop_pose[2]
    
    node_.get_logger().info(f"   ✅ '{item_name}' 배치 완료 (Top-Down 방식)")
    
    return response
    
    

def main(args=None):
    global node_, dsr_control_node_, vision_cli, gripper, gripper_manager, slot_manager
    # ========== 추가 ==========
    global cli_move_pause, cli_move_resume
    # ==========================
    
    rclpy.init(args=args)
    
    node_ = rclpy.create_node("item_placement_controller", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_item_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_
    
    load_calibration()
    
    slot_manager = SlotManager()
    
    node_.get_logger().info("=========================================")
    node_.get_logger().info("🔧 Item Placement Controller Node (Updated)")
    node_.get_logger().info("=========================================")
    
    try:
        gripper = RG("rg2", "192.168.1.1", "502")
        node_.get_logger().info("✅ Real Gripper Initialized")
    except:
        gripper = RG()
        node_.get_logger().info("⚠️ Virtual Gripper Initialized")
    
    gripper_manager = GripperManager(gripper)
    
    cb_group = ReentrantCallbackGroup()
    node_.create_service(PlaceItem, '/place_item', handle_place_item, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)
    
    # ========== 추가 ==========
    # Force Monitor 구독
    node_.create_subscription(
        Float64MultiArray,  # ← 타입 변경
        f'/{ROBOT_ID}/msg/tool_force',
        force_callback,
        10,
        callback_group=cb_group
    )
    
    # Pause/Resume 클라이언트
    cli_move_pause = node_.create_client(
        MovePause,
        f'/{ROBOT_ID}/motion/move_pause',
        callback_group=cb_group
    )
    cli_move_resume = node_.create_client(
        MoveResume,
        f'/{ROBOT_ID}/motion/move_resume',
        callback_group=cb_group
    )
    
    node_.get_logger().info("🛡️ Force Monitor Integrated")
    # ==========================
    
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