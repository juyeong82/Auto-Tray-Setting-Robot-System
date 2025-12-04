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
import math
import threading
from std_msgs.msg import Int32  # 속도 제어 메시지 추가

import DR_init
from ff_robot_interfaces.srv import OrderService, DetectObject, PlaceItem
from ff_robot.order_logic import SlotManager
from ff_robot.tray_manager import TrayManager

# ========== 기존 추가한 부분을 아래로 교체 ==========
from std_msgs.msg import Float64MultiArray
from dsr_msgs2.srv import MovePause, MoveResume
# =================================================

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================

# [속도 제어용 전역 변수]
g_current_speed = 100

GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0

TRAY_PICK_X_OFFSET = 50.0
TRAY_PICK_Z_OFFSET = 0.0

# 12/02/18:12 수정됨: 트레이 보충 시 집는 높이가 높아 값을 낮춤 (25.0 -> 15.0)
# 필요에 따라 20.0 ~ 23.0 사이로 조절하세요.
TRAY_PICK_HEIGHT_Z = 15.0

SERVING_PICK_OFFSET_Y = 120.0
SERVING_PICK_Z_OFFSET = 12.0  # ⬆️ 변경: -12.0 → 12.0
SERVING_GRIP_RZ = 180.0

SERVING_PUSH_DISTANCE = 300.0  # ⬆️ 변경: 150.0 → 250.0

# 12/02/18:35 수정됨: 서빙 시 충돌 방지를 위해 그리퍼를 조금만(30mm) 열도록 설정 (단위: 1/10mm)
SERVING_OPEN_WIDTH = 300

TRAY_CENTER_OFFSET_X = 80.0
TRAY_0_EDGE_POS = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
TRAY_1_EDGE_POS = [435.0, 20.0, 25.0, 43.35, -180.0, -134.82]


J_TRAY_PUSH_0 = [-23.22, 0.82, 121.41, -0.02, 57.77, 158.21]
J_TRAY_PUSH_1 = [-14.07, 30.29, 82.65, 0.03, 66.89, 167.03]

TRAY_FLOOR_Z = -20.0  # ⬆️ 변경: 25.0 → -25.0
# SAFE_Z_FLOOR_LIMIT = 0

APPROACH_HEIGHT = 100.0
EXTRA_LIFT_HEIGHT = 50.0

# [Joint Positions]
J_TRAY_OBSERVE = [0.0, 30.0, 25.0, 0.0, 110.0, 0.0]
# J_ITEM_OBSERVE = [-34.0, 33.0, 15.0, 0.0, 132.0, 144.0]
J_ITEM_OBSERVE = [-33.197, 21.512, 35.707, -0.118, 122.787, 144.296]

# ⭐ [NEW] 충돌 회피 경로 - Slot 0 보충용
J_AVOID_PATH_1 = [4.50, 15.38, 50.94, -0.30, 90.49, 1.31]
J_AVOID_PATH_2 = [1.44, -30.59, 91.91, -0.40, 99.27, 1.31]


ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# ========== 추가 ==========
# [Force Monitor Config]
FORCE_THRESHOLD = 25.0  # N
MOVING_AVG_WINDOW = 5
COOLDOWN_TIME = 1.0  # seconds
# ==========================

node_ = None
dsr_control_node_ = None
manager = None
tray_manager = None
vision_cli = None
place_item_cli = None
gripper = None

# ========== 추가 ==========
cli_move_pause = None
cli_move_resume = None
is_paused = False
force_buffer = []
last_trigger_time = 0.0
# ==========================

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

def transform_camera_to_base(cam_xyz):
    from DSR_ROBOT2 import get_current_posx
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
        if os.path.exists(npy_path):
            T_GRIPPER_TO_CAM = np.load(npy_path)
        else:
            return None
            
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
    # if gripper:
    #     try:
    #         gripper.open_gripper()
    #     except Exception as e:
    #         node_.get_logger().warn(f"⚠️ Gripper open failed (continuing): {e}")
    
    # 
    if gripper:
        try:
            gripper.move_gripper(SERVING_OPEN_WIDTH)
            time.sleep(0.5)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper move failed (continuing): {e}")
    
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
# [Action] 서빙 (뒤에서 밀기 - Pushing Motion) - 555 : 12.03 서빙 완료 후 모션 수정
# ==============================================================================
def serve_tray(slot_id):
    global tray_manager, gripper, SERVING_PUSH_DISTANCE, APPROACH_HEIGHT
    
    # 1. 트레이 배치 위치(작업대 위)의 중심 좌표를 가져옵니다.
    center_pos = get_tray_center_pose(slot_id)
    
    # 2. 푸시 시작점 계산 (상공 접근용 좌표만 계산)
    push_start_offset_y = 150.0 
    push_contact_pos = list(center_pos)
    push_contact_pos[1] -= push_start_offset_y 
    push_contact_pos[5] = SERVING_GRIP_RZ
    
    # 3. 그리퍼 열기 (그리퍼 밑판을 푸시 툴로 사용)
    if gripper:
        try:
            node_.get_logger().info("   👐 그리퍼 열기 (푸시 준비)")
            gripper.open_gripper() 
            time.sleep(0.5) 
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper open failed: {e}")
            
    # 4. 접근 위치 (상공) - 기존 유지 (안전을 위해 위로 먼저 이동)
    approach_push = list(push_contact_pos)
    approach_push[2] += APPROACH_HEIGHT # 안전한 높이로 접근
    if not safe_movel(approach_push, "서빙 (뒤) 상공 접근"): return False

    # =========================================================================
    # [수정됨] 5. 푸시 시작 위치로 하강 (Joint 이동 사용)
    # =========================================================================
    # 좌표 계산(TRAY_FLOOR_Z 등)을 쓰지 않고, 티칭된 관절 각도로 바로 이동하여 Z축을 고정합니다.
    # target_joint = J_TRAY_PUSH_0 if slot_id == 0 else J_TRAY_PUSH_1
    
    # node_.get_logger().info(f"   ⬇️ 서빙 높이 하강 (Joint 이동): Slot {slot_id}")
    
    # # safe_movej를 사용하여 관절 이동 (특이점 회피 및 Z축 높이 확정)
    # if not safe_movej(target_joint): 
    #     node_.get_logger().error("   ❌ 서빙 하강 실패 (MoveJ)")
    #     return False

    # [복구됨] 좌표(Z)로 하강하는 로직
    # TRAY_FLOOR_Z는 설정값(현재 -15.0)을 사용합니다.
    push_pose = list(push_contact_pos)
    push_pose[2] = TRAY_FLOOR_Z

    node_.get_logger().info(f"   ⬇️ 서빙 높이 하강 (Movel): Z={TRAY_FLOOR_Z}")
    if not safe_movel(push_pose, "서빙 (뒤) 접촉 하강"): return False
    # =========================================================================
    # [수정됨] 6. 푸시 실행 (현재 Joint 위치 기준 Y축 이동)
    # =========================================================================
    from DSR_ROBOT2 import get_current_posx
    
    # 방금 Joint로 이동한 '실제 로봇 위치'를 가져옵니다.
    curr_pos = get_current_posx()
    if curr_pos is None: return False
    
    # API 버전에 따라 튜플((pos, sol))로 올 수 있으므로 처리
    start_push_pose = list(curr_pos[0]) if isinstance(curr_pos, tuple) else list(curr_pos)
    
    # 목표 위치 계산 (현재 위치에서 Y축만 증가시킴)
    final_push_pose = list(start_push_pose)
    final_push_pose[1] += SERVING_PUSH_DISTANCE 
    
    node_.get_logger().info(f"   🚀 서빙 밀기 시작 (+Y {SERVING_PUSH_DISTANCE}mm)")
    
    # Z축은 Joint 이동으로 맞춰진 높이가 그대로 유지됨 (Linear Move)
    if not safe_movel(final_push_pose, "트레이 밀기 동작"): return False

    # 7. 안전하게 상승하여 복귀
    depart_pose = list(final_push_pose)
    depart_pose[2] += APPROACH_HEIGHT
    
    if not safe_movel(depart_pose, "서빙 완료 후 상승"): return False
    
    node_.get_logger().info(f"   ✅ 서빙 완료!")
    
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
    
    node_.get_logger().info("✅ 초기 상태 설정: 트레이 2개 세팅 완료 (Status -> Working)")
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

            # =================================================================
            # [수정] Phase 2: 아이템 배치 (재고 유무에 따른 유동적 순서 처리)
            # =================================================================
            needed_items = manager.get_all_needed_items()
            
            # 1. 중복 제거 (순서 유지: FIFO)
            # 예: ['burger', 'burger', 'coke'] -> ['burger', 'coke']
            unique_needed_items = []
            for x in needed_items:
                if x not in unique_needed_items:
                    unique_needed_items.append(x)

            target_item = None
            
            # [핵심 수정] 재고를 확인하려면 먼저 '관측 위치'로 가서 봐야 합니다!
            if unique_needed_items:
                # 서빙하고 돌아왔거나 딴청 피우고 있을 수 있으니 강제로 이동
                safe_movej(J_ITEM_OBSERVE)
                time.sleep(0.5) # 카메라 흔들림 안정화

                # 리스트를 순회하며 "지금 비전으로 보이는 것"을 찾음
                for item_candidate in unique_needed_items:
                    # 비전 서비스 호출
                    vision_check = call_vision_service(item_candidate)
                    
                    if vision_check is not None:
                        target_item = item_candidate
                        if item_candidate != unique_needed_items[0]:
                            node_.get_logger().info(f"🔀 [우선순위 변경] '{unique_needed_items[0]}' 부재 -> '{target_item}' 먼저 처리")
                        break
                    else:
                        pass

            # 3. 작업 수행 (타겟이 결정된 경우)
            if target_item:
                target_slot = manager.peek_target_slot(target_item)

            # 3. 작업 수행 (타겟이 결정된 경우)
            if target_item:
                target_slot = manager.peek_target_slot(target_item)
                
                # 유효성 검사 (해당 슬롯이 working 상태여야 함)
                if tray_manager.tray_states[target_slot]['status'] == 'working':
                    slot_data = manager.active_slots[target_slot]
                    current_idx = len(slot_data['placed_total'])
                    total_count = len(slot_data['needed'])
                    
                    # Module 1에게 작업 요청
                    if call_place_item_service(target_item, target_slot, current_idx, total_count):
                        is_done = manager.mark_item_done(target_slot, target_item)
                        if is_done:
                            tray_manager.update_tray_status(target_slot, "ready")
                else:
                    time.sleep(0.5)
            else:
                # 필요한 건 있는데 아무것도 안 보임 -> 대기
                if needed_items:
                    # 로그 너무 많이 뜨지 않게 1초 대기
                    # node_.get_logger().info(f"⏳ 재고 대기 중... (필요: {unique_needed_items})")
                    time.sleep(1.0)
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

# ------------------------------------------------------------------------------
# [Speed Control] 속도 제어 로직 (스레드 처리)
# ------------------------------------------------------------------------------
def process_speed_change_task(target_speed):
    global g_current_speed
    from DSR_ROBOT2 import change_operation_speed

    if target_speed > g_current_speed + 30:
        node_.get_logger().warn(f"📈 가속 Ramping: {g_current_speed} -> {target_speed}")
        temp_speed = g_current_speed
        while temp_speed < target_speed:
            temp_speed += 20
            if temp_speed > target_speed: temp_speed = target_speed
            change_operation_speed(temp_speed)
            time.sleep(0.2)
    else:
        change_operation_speed(target_speed)

    g_current_speed = target_speed
    node_.get_logger().warn(f"⚡ [속도 변경 완료] {target_speed}%")

def speed_callback(msg):
    target_speed = msg.data
    if 0 < target_speed <= 100:
        t = threading.Thread(target=process_speed_change_task, args=(target_speed,), daemon=True)
        t.start()
    else:
        node_.get_logger().warn(f"⚠️ 잘못된 속도 값: {target_speed}")

def main(args=None):
    global node_, dsr_control_node_, manager, tray_manager, vision_cli, place_item_cli
    # ========== 추가 ==========
    global cli_move_pause, cli_move_resume
    # ==========================
    rclpy.init(args=args)
    
    manager = SlotManager()
    tray_manager = TrayManager()
    
    node_ = rclpy.create_node("order_orchestrator", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_orchestrator_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_
    
    node_.get_logger().info("=========================================")
    node_.get_logger().info("🎯 Order Orchestrator Node (Updated)")
    node_.get_logger().info("=========================================")
    
    cb_group = ReentrantCallbackGroup()
    node_.create_service(OrderService, '/dsr01/order_service', handle_order_request, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)
    place_item_cli = node_.create_client(PlaceItem, '/place_item', callback_group=cb_group)
    
    # ========== 추가 ==========
    # Force Monitor 구독
    node_.create_subscription(
        Float64MultiArray,  # ← 타입 변경
        f'/{ROBOT_ID}/msg/tool_force',
        force_callback,
        10,
        callback_group=cb_group
    )

    # [추가] 속도 제어 토픽 구독
    node_.create_subscription(
        Int32, 
        '/robot_speed', 
        speed_callback, 
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