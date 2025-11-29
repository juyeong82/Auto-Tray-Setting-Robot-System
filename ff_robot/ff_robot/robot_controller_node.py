#!/usr/bin/env python3
# robot_controller_node_gem.py (Speed 50 + Waypoint Fix)

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

# [설정] 로그 즉시 출력
sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# [🎛️ 사용자 튜닝 섹션]
# ==============================================================================

# 1. [속도] 전체 로봇 속도/가속도 설정 (50으로 감속)
ROBOT_VEL = 20.0
ROBOT_ACC = 20.0

# 2. [경유지] 특이점 회피를 위한 중간 경유 자세 (JReady / Home)
# 비전 인식 후, 이 자세를 거쳐서 Pick 위치로 내려갑니다.
J_INTERMEDIATE_POS = [-23.82, 1.08, 91.17, -18.38, 80.61, -23.36] 

# 3. 픽 오프셋 (기존 값 유지)
GLOBAL_OFFSET_X = -5.0     
GLOBAL_OFFSET_Y = 120.0   
GLOBAL_OFFSET_Z = 110.0   

# 4. 아이템별 추가 오프셋
ITEM_OFFSETS = {
    "burger1": {"x": 0.0, "y": 0.0, "z": 10.0, "rx": -30.0, "ry": 180.0, "rz": 0.0}, 
    "burger2": {"x": 0.0, "y": 0.0, "z": 10.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "burger3": {"x": 0.0, "y": 0.0, "z": 10.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "coke":    {"x": -5.0, "y": -10.0, "z": 35.0}, 
    "cider":   {"x": -5.0, "y": -10.0, "z": 35.0},
    "fries":   {"x": 5.0, "y": 0.0, "z": 0.0},
    "nugget":  {"x": 5.0, "y": 0.0, "z": 0.0},
}

# 5. 접근 높이
APPROACH_HEIGHT = 50.0 

# 6. 관측 자세 (수정된 값 적용)
J_LOOK_POS = [40.05, 5.26, 58.84, -44.83, 133.16, -165.55]

# 7. 임시 Place 위치
TEMP_PLACE_POS = [300.0, 10.0, 200.0, 0.0, 180.0, 0.0]

# 8. 안전 바닥 높이
SAFE_Z_FLOOR_LIMIT = 10.0 
# ==============================================================================

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# --- 전역 변수 ---
node_ = None          
dsr_control_node_ = None 
manager = None
vision_cli = None 
gripper = None
gripper_manager = None
T_GRIPPER_TO_CAM = None

# ==============================================================================
# [설정 1] 캘리브레이션
# ==============================================================================
def load_calibration():
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    if os.path.exists(npy_path):
        T_GRIPPER_TO_CAM = np.load(npy_path)
        print(f"✅ 캘리브레이션 로드: {npy_path}")
    else:
        print(f"❌ [CRITICAL] 캘리브레이션 파일 없음!")
        sys.exit(1)

# ==============================================================================
# [설정 2] 가상 그리퍼
# ==============================================================================
try:
    from ff_robot.onrobot import RG
    print("✅ Real Gripper Driver Loaded")
except ImportError:
    class RG:
        def __init__(self, *args): pass
        def open_gripper(self): print("   👐 [Virtual] Open")
        def close_gripper(self, force=None): print("   ✊ Close")
        def move_gripper(self, width): print(f"   👌 [Virtual] Move Width: {width}")

# ==============================================================================
# [Math] 좌표 변환
# ==============================================================================
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
        print(f"   ❌ 변환 에러: {e}")
        return None

# ==============================================================================
# [Motion] 안전 이동 함수 (속도 50 적용)
# ==============================================================================
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
        # [수정] 전역 속도 변수 사용 (50.0)
        movel(pos, vel=[ROBOT_VEL, ROBOT_VEL], acc=[ROBOT_ACC, ROBOT_ACC], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05) 
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

# ==============================================================================
# [Action] Pick 동작
# ==============================================================================
def safe_move_and_pick(base_pos, item_name):
    global gripper_manager, node_
    
    # 1. 오프셋 정보 가져오기
    specific_offset = ITEM_OFFSETS.get(item_name, {})
    
    # 위치 오프셋 적용
    target_x = base_pos[0] + GLOBAL_OFFSET_X + specific_offset.get("x", 0.0)
    target_y = base_pos[1] + GLOBAL_OFFSET_Y + specific_offset.get("y", 0.0)
    target_z = base_pos[2] + GLOBAL_OFFSET_Z + specific_offset.get("z", 0.0)
    
    # [수정] 회전 오프셋 적용 (없으면 기본값 0, 180, 0 사용)
    target_rx = specific_offset.get("rx", 0.0)
    target_ry = specific_offset.get("ry", 180.0)
    target_rz = specific_offset.get("rz", 0.0)

    # 로그 출력
    if specific_offset:
        node_.get_logger().info(f"   🔧 [{item_name}] 보정: {specific_offset}")

    # 안전장치 (Z값 제한)
    if target_z < SAFE_Z_FLOOR_LIMIT:
        node_.get_logger().warn(f"⚠️ Z값이 너무 낮음({target_z:.1f}) -> {SAFE_Z_FLOOR_LIMIT}mm로 제한")
        target_z = SAFE_Z_FLOOR_LIMIT

    # 2. 최종 자세 결합 (설정된 rx, ry, rz 적용)
    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    
    # 접근(Approach) 좌표: 위치는 같고 높이만 다름 (회전은 Pick과 동일하게 유지)
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    
    node_.get_logger().info(f"   🎯 최종 Pick: {pick_pose}")

    # --- 동작 시퀀스 ---
    
    # (1) 접근 (기울어진 상태로 접근)
    if not safe_movel(approach_pose, "접근"): return False
    
    # (2) 그리퍼 준비
    if gripper_manager:
        gripper_manager.prepare_grip(item_name)

    # (3) 하강
    if not safe_movel(pick_pose, "하강"): return False
    
    # (4) 집기
    if gripper_manager:
        gripper_manager.execute_grip(item_name)
    else:
        if gripper: gripper.close_gripper()
        time.sleep(0.5)

    # (5) 상승 (그대로 올라옴)
    if not safe_movel(approach_pose, "상승"): return False
    
    return True

# ==============================================================================
# [Task Thread] 로봇 업무 로직
# ==============================================================================
def call_vision_service(target_name):
    global node_, vision_cli
    
    print(f"   [Debug] 비전 호출: {target_name}")
    if vision_cli is None or not vision_cli.service_is_ready():
        return None
    
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 10.0: 
            print("   ⚠️ [Timeout] Vision 응답 없음")
            return None
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
    except Exception as e:
        node_.get_logger().warn(f"⚠️ Real Gripper Fail: {e}")
        gripper = RG()

    gripper_manager = GripperManager(gripper)

    # 초기 이동 (관측)
    node_.get_logger().info(f"🔭 관측 위치 이동... {J_LOOK_POS}")
    try:
        movej(J_LOOK_POS, vel=ROBOT_VEL, acc=ROBOT_ACC)
        wait_for_motion()
    except Exception as e:
        node_.get_logger().error(f"❌ 초기 이동 실패: {e}")
    
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

            # 1. 관측 위치 복귀
            try:
                movej(J_LOOK_POS, vel=ROBOT_VEL, acc=ROBOT_ACC)
                wait_for_motion()
            except: pass

            if gripper_manager:
                gripper_manager.prepare_grip(target_item)

            # 2. 비전
            cam_xyz = call_vision_service(target_item)
            if cam_xyz is None:
                node_.get_logger().warn("   ⚠️ 타겟 못 찾음")
                time.sleep(1.0)
                continue

            # 3. 변환
            base_xyz = transform_camera_to_base(cam_xyz)
            if base_xyz is None:
                time.sleep(1.0)
                continue

            # ========================================================
            # [핵심 수정] 중간 경유지 이동 (특이점 회피)
            # 관측 자세 -> 중간 자세(movej) -> 픽 접근(movel)
            # ========================================================
            node_.get_logger().info(f"   🛡️ 특이점 회피: 중간 경유지 이동...")
            try:
                movej(J_INTERMEDIATE_POS, vel=ROBOT_VEL, acc=ROBOT_ACC)
                wait_for_motion()
            except Exception as e:
                node_.get_logger().error(f"❌ 경유지 이동 실패: {e}")
                continue

            # 4. Pick
            if safe_move_and_pick(base_xyz, target_item):
                node_.get_logger().info("✅ Pick 성공!")
                
                # 5. Place
                node_.get_logger().info(f"🚚 Place 이동...")
                
                # Place 하러 갈 때도 안전하게 경유지 들름 (선택 사항)
                # movej(J_INTERMEDIATE_POS, vel=ROBOT_VEL, acc=ROBOT_ACC)
                # wait_for_motion()

                if safe_movel(TEMP_PLACE_POS, "Place"):
                    if gripper_manager:
                        gripper_manager.release(target_item)
                    manager.clear_slot(0) 
                    node_.get_logger().info("🎉 작업 완료.")
            else:
                node_.get_logger().error("❌ Pick 실패")

            time.sleep(0.5)

        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

# ==============================================================================
# [Main]
# ==============================================================================
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