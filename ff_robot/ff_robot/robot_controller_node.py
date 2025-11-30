#!/usr/bin/env python3
# robot_controller_node_gem.py (Calibration Tuning Ver)

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
# [🎛️ 캘리브레이션 튜닝 섹션] (여기를 조절하세요!)
# ==============================================================================
# "물체 위치가 바뀌면 못 잡는 문제"는 여기서 해결해야 합니다.
# 카메라와 그리퍼 사이의 거리를 미세 조정합니다. (단위: mm)
# 이 값을 바꾸면 화면상 모든 물체의 좌표가 일괄적으로 이동합니다.

CALIB_FIX_X = 0.0   # 카메라가 생각보다 앞/뒤에 달려있다면 수정
CALIB_FIX_Y = 0.0   # 카메라가 생각보다 좌/우에 달려있다면 수정
CALIB_FIX_Z = 0.0   # (보통 건드릴 필요 없음)

# ==============================================================================
# [🎛️ 아이템별 튜닝] (Z축 높이와 회전만 건드리는 것을 추천)
# ==============================================================================
# X, Y는 위 CALIB_FIX로 잡고, 여기서는 0.0으로 두세요.

GLOBAL_OFFSET_X = 0.0     
GLOBAL_OFFSET_Y = -15.0   
GLOBAL_OFFSET_Z = -50.0   # 기본 픽 높이

ITEM_OFFSETS = {
    # X, Y는 0.0으로 초기화 권장
    "burger1": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": -30.0, "ry": 180.0, "rz": 0.0}, 
    "burger2": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "burger3": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "coke":    {"x": 0.0, "y": 0.0, "z": 40.0}, 
    "cider":   {"x": 0.0, "y": 0.0, "z": 40.0},
    "fries":   {"x": 0.0, "y": 0.0, "z": 0.0},
    "nugget":  {"x": 0.0, "y": 0.0, "z": 0.0},
}

# 접근 높이
APPROACH_HEIGHT = 100.0 
# 관측 자세
J_LOOK_POS = [40.05, 5.26, 58.84, -44.83, 133.16, -165.55]
# 임시 Place
TEMP_PLACE_POS = [300.0, 10.0, 400.0, 0.0, 180.0, 0.0]
# 안전 높이
SAFE_Z_FLOOR_LIMIT = 10.0 

# --- 로봇 설정 ---
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

# ==============================================================================
# [핵심] 캘리브레이션 로드 및 보정 적용
# ==============================================================================
def load_calibration():
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    
    if os.path.exists(npy_path):
        T_base = np.load(npy_path)
        
        # [수정] 캘리브레이션 미세 조정값 적용
        # T_gripper2camera 행렬의 이동(Translation) 부분에 더함
        # 주의: 이 좌표계는 "그리퍼 기준"입니다. (로봇 베이스 기준 아님)
        T_base[0, 3] += (CALIB_FIX_X / 1000.0) # mm -> m 변환
        T_base[1, 3] += (CALIB_FIX_Y / 1000.0)
        T_base[2, 3] += (CALIB_FIX_Z / 1000.0)
        
        T_GRIPPER_TO_CAM = T_base
        print(f"✅ 캘리브레이션 로드 및 보정 완료: {npy_path}")
        print(f"   👉 적용된 보정값(mm): X={CALIB_FIX_X}, Y={CALIB_FIX_Y}, Z={CALIB_FIX_Z}")
    else:
        print(f"❌ [CRITICAL] 캘리브레이션 파일 없음!")
        sys.exit(1)

# ... (RG, Math 함수 기존 동일) ...
try:
    from ff_robot.onrobot import RG
    print("✅ Real Gripper Driver Loaded")
except ImportError:
    class RG:
        def __init__(self, *args): pass
        def open_gripper(self): print("   👐 [Virtual] Open")
        def close_gripper(self, force=None): print("   ✊ Close")
        def move_gripper(self, width): print(f"   👌 [Virtual] Move Width: {width}")

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
        
        # 여기서 이미 보정된 T_GRIPPER_TO_CAM을 사용함
        T_base_cam = T_base_gripper @ T_GRIPPER_TO_CAM
        
        p_cam = np.array([cam_xyz[0]*1000, cam_xyz[1]*1000, cam_xyz[2]*1000, 1.0])
        p_base = T_base_cam @ p_cam
        return p_base[:3]
    except Exception as e:
        print(f"   ❌ 변환 에러: {e}")
        return None

def wait_for_motion():
    from DSR_ROBOT2 import check_motion
    time.sleep(0.1)
    while check_motion() != 0:
        time.sleep(0.05) 
        if not rclpy.ok(): return False
    return True

def safe_movel(pos, desc="이동"):
    global node_
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    try:
        time.sleep(0.05) 
        movel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05) 
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

def safe_move_and_pick(base_pos, item_name):
    global gripper_manager, node_
    
    # 1. 오프셋 계산 (공통 + 개별)
    specific_offset = ITEM_OFFSETS.get(item_name, {"x":0.0, "y":0.0, "z":0.0})
    
    target_x = base_pos[0] + GLOBAL_OFFSET_X + specific_offset.get("x", 0.0)
    target_y = base_pos[1] + GLOBAL_OFFSET_Y + specific_offset.get("y", 0.0)
    target_z = base_pos[2] + GLOBAL_OFFSET_Z + specific_offset.get("z", 0.0)
    
    target_rx = specific_offset.get("rx", 0.0)
    target_ry = specific_offset.get("ry", 180.0)
    target_rz = specific_offset.get("rz", 0.0)

    if specific_offset["x"] != 0:
        node_.get_logger().info(f"   🔧 [{item_name}] 보정: {specific_offset}")

    if target_z < SAFE_Z_FLOOR_LIMIT:
        target_z = SAFE_Z_FLOOR_LIMIT

    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    
    node_.get_logger().info(f"   🎯 최종 Pick: {pick_pose}")

    if not safe_movel(approach_pose, "접근"): return False
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    if not safe_movel(pick_pose, "하강"): return False
    
    if gripper_manager: gripper_manager.execute_grip(item_name)
    else:
        if gripper: gripper.close_gripper()
        time.sleep(0.5)

    if not safe_movel(approach_pose, "상승"): return False
    return True

# ... (call_vision_service, perform_task 등 로직 동일) ...
def call_vision_service(target_name):
    global node_, vision_cli
    print(f"   [Debug] 비전 호출: {target_name}")
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
        if res.found: return [res.position.x, res.position.y, res.position.z]
    except: pass
    return None

def perform_robot_task():
    global node_, manager, gripper, gripper_manager
    try: from DSR_ROBOT2 import movej
    except: return

    try:
        gripper = RG("rg2", "192.168.1.1", "502") 
        node_.get_logger().info("✅ Real Gripper Connected")
    except:
        gripper = RG()

    gripper_manager = GripperManager(gripper)

    # 초기 이동
    node_.get_logger().info(f"🔭 관측 위치 이동... {J_LOOK_POS}")
    try:
        movej(J_LOOK_POS, vel=40.0, acc=40.0)
        wait_for_motion()
    except: pass
    
    node_.get_logger().info("[Task] 준비 완료.")

    while rclpy.ok():
        try:
            needed_items = manager.get_all_needed_items()
            if not needed_items:
                time.sleep(1.0) 
                continue

            target_item = needed_items[0]
            node_.get_logger().info(f"========================================")
            node_.get_logger().info(f"🔎 [Search] '{target_item}'")

            try:
                movej(J_LOOK_POS, vel=60.0, acc=60.0)
                wait_for_motion()
                # 안정화 대기
                time.sleep(1.0)
            except: pass

            if gripper_manager: gripper_manager.prepare_grip(target_item)

            cam_xyz = call_vision_service(target_item)
            if cam_xyz is None:
                node_.get_logger().warn("   ⚠️ 타겟 못 찾음 (Retry)")
                time.sleep(1.0)
                continue

            base_xyz = transform_camera_to_base(cam_xyz)
            if base_xyz is None:
                time.sleep(1.0)
                continue

            if safe_move_and_pick(base_xyz, target_item):
                node_.get_logger().info("✅ Pick 성공!")
                
                node_.get_logger().info(f"🚚 Place 이동...")
                if safe_movel(TEMP_PLACE_POS, "Place"):
                    if gripper_manager:
                        gripper_manager.release(target_item)
                    
                    # 완료 처리
                    manager.clear_slot(0) 
                    node_.get_logger().info("🎉 작업 완료.")
            else:
                node_.get_logger().error("❌ Pick 실패")

            time.sleep(0.5)

        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

def handle_order_request(request, response):
    global manager, node_
    node_.get_logger().info(f"⚡ [Service] 주문: {request.item_names}")
    success, msg = manager.check_and_deduct_stock(request.item_names, request.item_quantities)
    if success:
        manager.add_order_to_slot(f"ORD-{int(time.time())}", request.item_names, request.item_quantities)
        response.assigned_order_id = f"ORD-{int(time.time())}"
        response.message = "접수 완료"
        node_.get_logger().info("✅ 주문 처리됨")
    else:
        response.assigned_order_id = ""
        response.message = msg
        node_.get_logger().warn(f"🚫 주문 거절: {msg}")
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