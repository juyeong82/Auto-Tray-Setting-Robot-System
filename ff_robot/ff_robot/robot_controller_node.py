#!/usr/bin/env python3
# robot_controller_node.py
# [Final Fix] 그리퍼 전역변수 연결 오류 수정 + Pick X 오프셋 추가

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

# --------------------------
# 🎛️ CONFIGURATION
# --------------------------
GLOBAL_OFFSET_X = 0.0     
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0     

# [NEW] 트레이 집을 때 오프셋 설정 (단위: mm)
# 관측된 위치보다 X축, Z축으로 더 이동해서 잡습니다.
TRAY_PICK_X_OFFSET = 50.0   # [수정] X축 방향 보정 (필요에 따라 +/- 조절)
TRAY_PICK_Z_OFFSET = -12  # Z축 방향 보정 (더 깊게 잡기)

ITEM_OFFSETS = {
    "burger1": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "burger2": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "burger3": {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "coke":    {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None}, 
    "cider":   {"z": 20.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "fries":   {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
    "nugget":  {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None},
}

SCALE_X_TARGETS = ['cider', 'coke', 'fries', 'nugget']
SCALE_Y_TARGETS = ['burger1', 'burger2', 'burger3']
SCALE_FACTOR_X = 1.05 
SCALE_FACTOR_Y = 1.05
TILT_FACTOR_X = 0.10

J_TRAY_OBSERVE = [0.0, 30.0, 25.0, 0.0, 110.0, 0.0]  
J_ITEM_OBSERVE = [45.0, 20.0, 30.0, 0.0, 130.0, 135.0]

APPROACH_HEIGHT = 100.0
SAFE_Z_FLOOR_LIMIT = -15.0 

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

# def wait_for_motion():
#     from DSR_ROBOT2 import check_motion
#     # [수정] 모션 시작 전 안정화 대기 시간을 조금 더 줍니다.
#     # 명령 직후에는 check_motion이 즉시 반응하지 않을 수 있음
    
#     # 1. 움직임이 감지될 때까지 잠깐 대기 (최대 1초)
#     wait_start = time.time()
#     while check_motion() == 0:
#         if time.time() - wait_start > 1.0: 
#             # 1초가 지났는데도 안 움직이면 진짜 안 움직이는 것이거나 이미 끝난 것
#             return True
#         time.sleep(0.05)
#         if not rclpy.ok(): return False
        
#     # 2. 움직임이 시작됨 -> 멈출 때까지 대기
#     while check_motion() != 0:
#         time.sleep(0.05)
#         if not rclpy.ok(): return False
        
#     return True

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
        node_.get_logger().info(f"   🏃 [{desc}] -> {pos[:3]}") 
        time.sleep(0.05) 
        movel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        if not wait_for_motion():
            node_.get_logger().error(f"   ❌ [{desc}] 모션 실패/타임아웃")
            return False
        time.sleep(0.05) 
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ [{desc}] 오류: {e}")
        return False

# ==============================================================================
# 물품 집기
# ==============================================================================
def safe_move_and_pick_item(base_pos, item_name, ref_x_pos, ref_y_pos, rot_rz):
    offset_info = ITEM_OFFSETS.get(item_name, {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None})
    raw_x, raw_y, raw_z = base_pos[0], base_pos[1], base_pos[2]

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
    target_rz = rot_rz if offset_info.get("rz") is None else offset_info["rz"]

    if target_z < SAFE_Z_FLOOR_LIMIT: target_z = SAFE_Z_FLOOR_LIMIT

    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    
    if not safe_movel(approach_pose, "아이템 접근"): return False
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    if not safe_movel(pick_pose, "아이템 하강"): return False
    if gripper_manager: gripper_manager.execute_grip(item_name)
    else:
        if gripper: gripper.close_gripper(); time.sleep(0.5)
    if not safe_movel(approach_pose, "아이템 상승"): return False
    return True

# ==============================================================================
# 트레이 집기
# ==============================================================================
def pick_and_place_tray(detected_data, slot_id):
    # [수정] gripper 변수를 전역으로 사용
    global tray_manager, gripper
    
    # 1. 트레이 정보 및 그립 좌표 계산
    cam_pos = detected_data["position"]
    base_pos = transform_camera_to_base(cam_pos) 
    rotation_rz = detected_data["rotation"][2] 
    
    if base_pos is None: return False

    # (1) TrayManager에서 기본 좌표 계산
    grip_pose = tray_manager.calculate_tray_grip_point(base_pos, rotation_rz)
    
    # (2) [수정] X, Z 오프셋 적용 (여기서 보정)
    grip_pose[0] += TRAY_PICK_X_OFFSET
    grip_pose[2] += TRAY_PICK_Z_OFFSET
    
    node_.get_logger().info(f"   🍱 Tray Grip Pose (Adjusted): {grip_pose}")
    
    # 2. 접근
    approach_pose = grip_pose[:]
    approach_pose[2] += APPROACH_HEIGHT
    
    # [수정] 그리퍼 열기 (None 체크)
    if gripper: 
        node_.get_logger().info("   👐 그리퍼 열기")
        gripper.open_gripper()
    
    if not safe_movel(approach_pose, "트레이 접근(상공)"): return False
    
    # [디버깅] 하강 전 현재 위치 확인
    try:
        from DSR_ROBOT2 import get_current_posx
        curr_before = get_current_posx()
        if isinstance(curr_before, tuple): curr_before = curr_before[0]
        node_.get_logger().info(f"   📍 [하강 전] 현재 위치: Z={curr_before[2]:.1f}mm")
    except: pass
    
    if not safe_movel(grip_pose, "트레이 잡기 위치 하강"): return False
    
    # [디버깅] 하강 후 현재 위치 확인
    try:
        from DSR_ROBOT2 import get_current_posx
        curr_after = get_current_posx()
        if isinstance(curr_after, tuple): curr_after = curr_after[0]
        node_.get_logger().info(f"   📍 [하강 후] 현재 위치: Z={curr_after[2]:.1f}mm")
        node_.get_logger().info(f"   📍 [목표 위치] Z={grip_pose[2]:.1f}mm")
        
        # 실제로 하강했는지 확인
        if abs(curr_after[2] - grip_pose[2]) > 20.0:
            node_.get_logger().error(f"   ❌ 하강 실패! 현재={curr_after[2]:.1f}, 목표={grip_pose[2]:.1f}")
            return False
    except Exception as e:
        node_.get_logger().warn(f"   ⚠️ 위치 확인 실패: {e}")
    
    # 3. 그립
    if gripper: 
        node_.get_logger().info("   ✊ 그리퍼 닫기")
        gripper.close_gripper()
        time.sleep(1.5)
    
    # 4. 배치 위치 계산 및 회전값 고정 (Drag 모드)
    place_pose = tray_manager.calculate_work_table_position(slot_id)
    place_pose[3] = grip_pose[3]
    place_pose[4] = grip_pose[4]
    place_pose[5] = grip_pose[5]
    
    node_.get_logger().info("   🚚 트레이 끄기(Slide - No Rotation)...")
    
    if not safe_movel(place_pose, "트레이 끌어서 이동"): return False
    
    # 5. 그리퍼 열기
    if gripper: 
        node_.get_logger().info("   👐 그리퍼 열기")
        gripper.open_gripper()
        time.sleep(1.0)
        
    # 6. 상승
    depart_pose = place_pose[:]
    depart_pose[2] += APPROACH_HEIGHT
    
    if not safe_movel(depart_pose, "작업 완료 후 상승"): return False
    
    return True

def serve_tray(slot_id):
    global tray_manager
    
    work_pose = tray_manager.calculate_work_table_position(slot_id)
    grip_pose = work_pose[:] 
    
    approach = grip_pose[:]
    approach[2] += APPROACH_HEIGHT
    
    if not safe_movel(approach, "서빙 준비 접근"): return False
    if not safe_movel(grip_pose, "서빙 그립"): return False
    if gripper: gripper.close_gripper(); time.sleep(0.5)
    if not safe_movel(approach, "서빙 들기"): return False
    
    serve_pose = tray_manager.calculate_serve_position(slot_id)
    serve_approach = serve_pose[:]
    serve_approach[2] += APPROACH_HEIGHT
    
    if not safe_movel(serve_approach, "서빙 위치 접근"): return False
    if not safe_movel(serve_pose, "서빙"): return False
    if gripper: gripper.open_gripper(); time.sleep(0.5)
    if not safe_movel(serve_approach, "완료"): return False
    
    return True

def call_vision_service(target_name):
    if vision_cli is None or not vision_cli.service_is_ready(): return None
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 10.0: return None
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

def perform_robot_task():
    # [수정] global gripper 추가 (이게 없어서 그리퍼가 안 됐음)
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
            # 1. 빈 슬롯 채우기
            empty_slot = tray_manager.get_empty_slot()
            if empty_slot is not None and (manager.pending_queue or not manager.get_all_needed_items()):
                node_.get_logger().info(f"🔎 [Phase 1] 트레이 탐색 (Slot {empty_slot})")
                safe_movej(J_TRAY_OBSERVE)
                time.sleep(1.0)
                
                tray_data = call_vision_service("tray")
                if tray_data:
                    if pick_and_place_tray(tray_data, empty_slot):
                        if manager.pending_queue:
                            manager.clear_slot(empty_slot)
                        tray_manager.update_tray_status(empty_slot, "working")
                        # [수정] 트레이 배치 완료 후 관측 자세로 복귀
                        node_.get_logger().info("   ↩️  트레이 배치 완료, 관측 자세로 복귀")
                        safe_movej(J_TRAY_OBSERVE)
                        continue 

            # 2. 물품 배치
            needed_items = manager.get_all_needed_items()
            if needed_items:
                target_item = needed_items[0]
                node_.get_logger().info(f"🔎 [Phase 2] 물품 탐색: '{target_item}'")
                safe_movej(J_ITEM_OBSERVE)
                time.sleep(1.0)
                
                try:
                    curr_pos = get_current_posx()
                    if isinstance(curr_pos, tuple): curr_pos = curr_pos[0]
                    ref_x_pos, ref_y_pos = curr_pos[0], curr_pos[1]
                except: ref_x_pos, ref_y_pos = 0.0, 0.0

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
                                if data and target_item in data['needed'] and data['placed_total'].count(target_item) < data['needed'].count(target_item):
                                    target_slot = sid
                                    break
                            
                            if target_slot is not None:
                                place_pos = tray_manager.calculate_work_table_position(target_slot)
                                place_pos[2] += 20.0 
                                safe_movel(place_pos, "물품 배치")
                                if gripper_manager: gripper_manager.release(target_item)
                                
                                is_done = manager.mark_item_done(target_slot, target_item)
                                if is_done:
                                    tray_manager.update_tray_status(target_slot, "ready")
                        
            # 3. 서빙
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