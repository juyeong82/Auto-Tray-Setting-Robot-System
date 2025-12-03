#!/usr/bin/env python3
# order_orchestrator.py (Updated from robot_controller_node.py)
# 주문 관리, 트레이 픽업/서빙, 전체 워크플로우 조율
# [Updated] 
# 1. ENABLE_STARTUP_PLACEMENT 추가
# 2. 충돌 회피 경로 추가 (J_AVOID_PATH_1, J_AVOID_PATH_2)
# 3. 설정값 업데이트 (TRAY_FLOOR_Z, SERVING_PUSH_DISTANCE 등)

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
from ff_robot_interfaces.srv import OrderService, DetectObject, PlaceItem
from ff_robot.order_logic import SlotManager
from ff_robot.tray_manager import TrayManager

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================

# [설정] 로봇 시작 시 자동으로 트레이를 배치할지 여부
ENABLE_STARTUP_PLACEMENT = False

GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = 0.0

TRAY_PICK_X_OFFSET = 0.0
TRAY_PICK_Z_OFFSET = 0.0
TRAY_PICK_HEIGHT_Z = 25.0  # ⬆️ 변경: 20.0 → 25.0

SERVING_PICK_OFFSET_Y = 120.0
SERVING_PICK_Z_OFFSET = 40.0  # ⬆️ 변경: -12.0 → 12.0
SERVING_GRIP_RZ = 90.0

SERVING_PUSH_DISTANCE = 250.0  # ⬆️ 변경: 150.0 → 250.0

TRAY_0_POS = [205.0, 20.0, 20.0, 43.35, -180.0, -134.82]
TRAY_1_POS = [435.0, 20.0, 20.0, 43.35, -180.0, -134.82]

TRAY_CENTER_OFFSET_X = 80.0
TRAY_0_EDGE_POS = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
TRAY_1_EDGE_POS = [435.0, 20.0, 25.0, 43.35, -180.0, -134.82]

TRAY_FLOOR_Z = -25.0  # ⬆️ 변경: 25.0 → -25.0
SAFE_Z_FLOOR_LIMIT = 10.0

APPROACH_HEIGHT = 100.0
EXTRA_LIFT_HEIGHT = 50.0

# [Joint Positions]
J_TRAY_OBSERVE = [0.0, 30.0, 25.0, 0.0, 110.0, 0.0]
J_ITEM_OBSERVE = [-34.0, 33.0, 15.0, 0.0, 132.0, 144.0]

# ⭐ [NEW] 충돌 회피 경로 - Slot 0 보충용
J_AVOID_PATH_1 = [4.50, 15.38, 50.94, -0.30, 90.49, 1.31]
J_AVOID_PATH_2 = [1.44, -30.59, 91.91, -0.40, 99.27, 1.31]

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

node_ = None
dsr_control_node_ = None
manager = None
tray_manager = None
vision_cli = None
place_item_cli = None
gripper = None

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
        def close_gripper(self, force=None): print("   ✊ [Virtual] Close")

def get_robot_pose_matrix(posx_list):
    x, y, z, rx, ry, rz = posx_list
    rot = R.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T

# [수정 후] T_cam2base.npy 로드 (변수명도 직관적으로 변경)
T_CAM_TO_BASE = None  # 전역 변수

def load_calibration():
    global T_CAM_TO_BASE
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Eye-to-Hand 캘리브레이션 결과 파일
    npy_path = os.path.join(current_dir, "T_cam2base.npy") 
    
    if os.path.exists(npy_path):
        T_CAM_TO_BASE = np.load(npy_path)
        if node_: node_.get_logger().info(f"✅ Fixed Camera Calibration loaded: {npy_path}")
    else:
        if node_: node_.get_logger().error(f"❌ Calibration file not found: {npy_path}")
        return False
    return True

# [함수 교체] 로봇 움직임(get_current_posx) 제거 및 고정 행렬 사용
def transform_camera_to_base(cam_xyz):
    # 전역 변수 T_CAM_TO_BASE 사용
    if T_CAM_TO_BASE is None:
        if not load_calibration(): return None

    try:
        # 1. 카메라 좌표계 점 (x, y, z, 1) 생성 (단위: mm)
        p_cam = np.array([cam_xyz[0]*1000, cam_xyz[1]*1000, cam_xyz[2]*1000, 1.0])
        
        # 2. Base 좌표계로 변환 (단순 행렬 곱)
        # 공식: P_base = T_cam2base * P_cam
        p_base = T_CAM_TO_BASE @ p_cam
        
        return p_base[:3] # (x, y, z) 반환
    except Exception as e:
        if node_: node_.get_logger().error(f"Transformation Error: {e}")
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
        node_.get_logger().info(f"      → {desc} 시작: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
        time.sleep(0.05)
        movel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05)
        node_.get_logger().info(f"      ✓ {desc} 완료")
        return True
    except Exception as e:
        node_.get_logger().error(f"      ✗ {desc} 실패: {e}")
        return False

# ==============================================================================
# [Action] 트레이 집기 (옆구리 잡기 -X -> EDGE_POS로 배치)
# ==============================================================================
def pick_and_place_tray(detected_data, slot_id):
    global tray_manager, gripper
    
    node_.get_logger().info(f"   [Step 1/8] 좌표 변환 시작")
    cam_pos = detected_data["position"]
    base_pos = transform_camera_to_base(cam_pos)
    rotation_rz = detected_data["rotation"][2]
    
    if base_pos is None:
        node_.get_logger().error(f"   ❌ 좌표 변환 실패")
        return False

    node_.get_logger().info(f"   [Step 2/8] 집기 좌표 계산")
    grip_pose = tray_manager.calculate_tray_grip_point(base_pos, rotation_rz)
    grip_pose[0] += TRAY_PICK_X_OFFSET
    
    # Z축 강제 고정
    grip_pose[2] = TRAY_PICK_HEIGHT_Z + TRAY_PICK_Z_OFFSET
    
    node_.get_logger().info(f"   🍱 Tray Side Grip Pose: {grip_pose}")
    
    approach_pose = grip_pose[:]
    approach_pose[2] += APPROACH_HEIGHT
    
    node_.get_logger().info(f"   [Step 3/8] 그리퍼 열기")
    if gripper:
        try:
            gripper.open_gripper()
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper open failed (continuing): {e}")
    
    node_.get_logger().info(f"   [Step 4/8] 접근 위치로 이동")
    if not safe_movel(approach_pose, "트레이 접근(상공)"):
        node_.get_logger().error(f"   ❌ 접근 실패")
        return False
        
    node_.get_logger().info(f"   [Step 5/8] 하강")
    if not safe_movel(grip_pose, "트레이 잡기 위치 하강"):
        node_.get_logger().error(f"   ❌ 하강 실패")
        return False
    
    node_.get_logger().info(f"   [Step 6/8] 그리퍼 닫기")
    if gripper:
        try:
            gripper.close_gripper()
            time.sleep(1.0)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper close failed (continuing): {e}")
    
    node_.get_logger().info(f"   [Step 7/8] 들어올리기 (250mm)")
    lift_pose = grip_pose[:]
    lift_pose[2] = 250.0
    if not safe_movel(lift_pose, "트레이 높게 들기"):
        node_.get_logger().error(f"   ❌ 들기 실패")
        return False

    # ⭐ [NEW] 충돌 회피 경로 - Slot 0 보충 시
    node_.get_logger().info(f"   [Step 8/8] 배치 위치로 이동 (Slot {slot_id})")
    if slot_id == 0:
        node_.get_logger().info("🚧 [Slot 0] 충돌 방지 우회 경로 실행 (Slot 1 회피)")
        if not safe_movej(J_AVOID_PATH_1):
            node_.get_logger().error(f"   ❌ 회피경로1 실패")
            return False
        if not safe_movej(J_AVOID_PATH_2):
            node_.get_logger().error(f"   ❌ 회피경로2 실패")
            return False
    
    target_pos = TRAY_0_EDGE_POS if slot_id == 0 else TRAY_1_EDGE_POS
    
    place_pose = list(target_pos)
    place_pose[3] = grip_pose[3]
    place_pose[4] = grip_pose[4]
    place_pose[5] = grip_pose[5]
    
    air_pose = place_pose[:]
    air_pose[2] = 250.0
    
    node_.get_logger().info(f"   [Step 8a] 공중 이동")
    if not safe_movel(air_pose, "트레이 공중 이동"):
        node_.get_logger().error(f"   ❌ 공중 이동 실패")
        return False
        
    node_.get_logger().info(f"   [Step 8b] 배치 하강")
    if not safe_movel(place_pose, "트레이 배치 하강"):
        node_.get_logger().error(f"   ❌ 배치 하강 실패")
        return False
    
    node_.get_logger().info(f"   [Complete] 그리퍼 열고 복귀")
    if gripper:
        try:
            gripper.open_gripper()
            time.sleep(0.5)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper open failed (continuing): {e}")
    
    depart_pose = place_pose[:]
    depart_pose[2] += APPROACH_HEIGHT
    if not safe_movel(depart_pose, "작업 완료 상승"):
        node_.get_logger().error(f"   ❌ 복귀 실패")
        return False
    
    node_.get_logger().info(f"   ✅✅✅ 트레이 픽업 완료! ✅✅✅")
    return True

# ==============================================================================
# [Action] 서빙 (아래 잡고 밀기 -Y)
# ==============================================================================
def serve_tray(slot_id):
    global tray_manager, gripper
    
    center_pos = get_tray_center_pose(slot_id)
    
    # 아래쪽 잡기
    grip_pose = list(center_pos)
    grip_pose[1] -= SERVING_PICK_OFFSET_Y
    grip_pose[2] = TRAY_FLOOR_Z + SERVING_PICK_Z_OFFSET
    grip_pose[5] = SERVING_GRIP_RZ
    
    approach = list(grip_pose)
    approach[2] += APPROACH_HEIGHT
    
    node_.get_logger().info(f"   🍽️ 서빙 시작 (Slot {slot_id}) - 좌표: {grip_pose[:3]}")
    
    if gripper:
        try:
            gripper.open_gripper()
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper open failed (continuing): {e}")
    
    if not safe_movel(approach, "서빙 준비 접근"): return False
    if not safe_movel(grip_pose, "서빙 그립 하강"): return False
    
    if gripper:
        try:
            gripper.close_gripper()
            time.sleep(1.0)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper close failed (continuing): {e}")
    
    # 밀기
    push_pose = list(grip_pose)
    push_pose[1] += SERVING_PUSH_DISTANCE  # ⬆️ 변경: 150 → 250mm
    
    node_.get_logger().info(f"   🚀 서빙 밀기 (+Y {SERVING_PUSH_DISTANCE}mm)")
    
    if not safe_movel(push_pose, "서빙 밀기 동작"): return False
    
    if gripper:
        try:
            gripper.open_gripper()
            time.sleep(0.5)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper open failed (continuing): {e}")
    
    depart = list(push_pose)
    depart[2] += APPROACH_HEIGHT
    
    if not safe_movel(depart, "서빙 완료 후 상승"): return False
    
    return True

# ⭐ [NEW] 트레이 보충 함수
def ensure_tray_in_slot(slot_id):
    """특정 슬롯이 비어있으면 트레이를 가져다 놓음"""
    node_.get_logger().info(f"🔎 [Tray Refill] Slot {slot_id} 보충 시작")
    safe_movej(J_TRAY_OBSERVE)
    time.sleep(1.0)
    
    tray_data = call_vision_service("tray")
    if tray_data:
        if pick_and_place_tray(tray_data, slot_id):
            tray_manager.update_tray_status(slot_id, "working")
            node_.get_logger().info(f"✅ Slot {slot_id} 트레이 보충 완료")
            return True
    
    node_.get_logger().warn(f"⚠️ Slot {slot_id} 트레이 보충 실패")
    return False

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

def call_place_item_service(item_name, target_slot_id, item_index, total_items):
    if place_item_cli is None or not place_item_cli.service_is_ready():
        node_.get_logger().error("❌ PlaceItem service not available")
        return False
    
    req = PlaceItem.Request()
    req.item_name = item_name
    req.target_slot_id = target_slot_id
    req.item_index = item_index
    req.total_items = total_items
    
    future = place_item_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 60.0:
            node_.get_logger().error("❌ PlaceItem service timeout")
            return False
        time.sleep(0.1)
    
    try:
        res = future.result()
        if res.success:
            node_.get_logger().info(f"   ✅ PlaceItem 성공: {res.message}")
            return True
        else:
            node_.get_logger().error(f"   ❌ PlaceItem 실패: {res.message}")
            return False
    except Exception as e:
        node_.get_logger().error(f"   ❌ PlaceItem 예외: {e}")
        return False

# ==============================================================================
# MAIN TASK LOOP
# ==============================================================================
def perform_robot_task():
    global manager, tray_manager, gripper
    
    try:
        from DSR_ROBOT2 import movej, get_current_posx
    except:
        return

    try:
        gripper = RG("rg2", "192.168.1.1", "502")
        node_.get_logger().info("✅ Real Gripper Initialized")
    except:
        gripper = RG()
        node_.get_logger().info("⚠️ Virtual Gripper Initialized")

    safe_movej(J_TRAY_OBSERVE)
    
    # ⭐ [NEW] 시작 시 트레이 배치 로직
    if ENABLE_STARTUP_PLACEMENT:
        node_.get_logger().info("🚀 로봇 시작: 초기 트레이 배치 작업 수행")
        if tray_manager.tray_states[0]['status'] == 'empty':
            ensure_tray_in_slot(0)
        if tray_manager.tray_states[1]['status'] == 'empty':
            ensure_tray_in_slot(1)
    else:
        node_.get_logger().info("🚧 로봇 시작: 초기 트레이 배치 건너뜀 -> 상태 'working' 강제 설정")
        tray_manager.update_tray_status(0, "working")
        tray_manager.update_tray_status(1, "working")
    
    safe_movej(J_ITEM_OBSERVE)
    node_.get_logger().info("[Task] 초기화 완료. 주문 대기 중...")

    while rclpy.ok():
        try:
            # Phase 1: 트레이 보충 (서빙 후 비어있으면)
            for sid in [0, 1]:
                if tray_manager.tray_states[sid]['status'] == 'empty':
                    node_.get_logger().info(f"🔎 [Phase 1] 트레이 보충 (Slot {sid})")
                    ensure_tray_in_slot(sid)

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
                    
                    target_slot = None
                    for sid, data in manager.active_slots.items():
                        if data and target_item in data['needed'] and \
                           data['placed_total'].count(target_item) < data['needed'].count(target_item):
                            target_slot = sid
                            break
                    
                    if target_slot is not None:
                        slot_data = manager.active_slots[target_slot]
                        current_idx = len(slot_data['placed_total'])
                        total_count = len(slot_data['needed'])
                        
                        if call_place_item_service(target_item, target_slot, current_idx, total_count):
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
    global node_, dsr_control_node_, manager, tray_manager, vision_cli, place_item_cli
    
    rclpy.init(args=args)
    
    manager = SlotManager()
    tray_manager = TrayManager()
    
    node_ = rclpy.create_node("order_orchestrator", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_orchestrator_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_
    
    load_calibration()
    
    node_.get_logger().info("=========================================")
    node_.get_logger().info("🎯 Order Orchestrator Node (Updated)")
    node_.get_logger().info("=========================================")
    
    cb_group = ReentrantCallbackGroup()
    node_.create_service(OrderService, '/dsr01/order_service', handle_order_request, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)
    place_item_cli = node_.create_client(PlaceItem, '/place_item', callback_group=cb_group)
    
    node_.get_logger().info("⏳ Waiting for services...")
    
    timeout_sec = 10.0
    start_time = time.time()
    while not place_item_cli.service_is_ready():
        if time.time() - start_time > timeout_sec:
            node_.get_logger().warn("⚠️ /place_item service not available (continuing anyway)")
            break
        time.sleep(0.1)
    
    if place_item_cli.service_is_ready():
        node_.get_logger().info("✅ /place_item service connected")
    
    executor = MultiThreadedExecutor()
    executor.add_node(node_)
    
    t_robot = threading.Thread(target=perform_robot_task, daemon=True)
    t_robot.start()
    
    node_.get_logger().info("🚀 Main loop started")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()