#!/usr/bin/env python3
# robot_controller_main.py (Full Workflow Integration)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import numpy as np
import os
import sys

import DR_init
from ff_robot_interfaces.srv import OrderService, DetectObject
from ff_robot.order_logic import SlotManager
from ff_robot.gripper import GripperManager
from ff_robot.tray_manager import TrayManager

sys.stdout.reconfigure(line_buffering=True)

# ========================================
# [🎛️ 로봇 설정]
# ========================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# ========================================
# [🎛️ 관측 위치 (Joint 각도)]
# ========================================
# # [수정 필요] 트레이 관측 위치
# J_TRAY_OBSERVE = [40.05, 5.26, 58.84, -44.83, 133.16, -165.55]


# # [수정 필요] 물품 관측 위치  
# J_ITEM_OBSERVE = [45.0, 10.0, 60.0, -50.0, 130.0, -170.0]


# 트레이 관측 위치 (조인트 각도, degree)
J_TRAY_OBSERVE = [0.0, 30.0, 25.0, 0.0, 110.0, 0.0]

# 트레이 관측 위치 (Cartesian 좌표, mm/degree) - 참고용
# X_TRAY_OBSERVE = [597.26, 7.41, 362.71, 0.57, 185.0, 0.47]

# 물품 관측 위치 (조인트 각도, degree)
J_ITEM_OBSERVE = [45.0, 20.0, 30.0, 0.0, 130.0, 135.0]

# 물품 관측 위치 (Cartesian 좌표, mm/degree) - 참고용
# [TODO] 태스크 탭에서 X, Y, Z, Rx, Ry, Rz 값 확인 필요
# X_ITEM_OBSERVE = [?, ?, ?, ?, ?, ?]

# ========================================
# [🎛️ 캘리브레이션]
# ========================================
CALIB_FIX_X = 0.0
CALIB_FIX_Y = 0.0
CALIB_FIX_Z = 0.0

# ========================================
# [🎛️ 아이템별 오프셋]
# ========================================
GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = -15.0
GLOBAL_OFFSET_Z = -50.0

ITEM_OFFSETS = {
    "burger1": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "burger2": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "burger3": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": -30.0, "ry": 180.0, "rz": 0.0},
    "coke": {"x": 0.0, "y": 0.0, "z": 40.0},
    "cider": {"x": 0.0, "y": 0.0, "z": 40.0},
    "fries": {"x": 0.0, "y": 0.0, "z": 0.0},
    "nugget": {"x": 0.0, "y": 0.0, "z": 0.0},
}

APPROACH_HEIGHT = 100.0
SAFE_Z_FLOOR_LIMIT = 10.0

# ========================================
# 전역 변수
# ========================================
node_ = None
dsr_control_node_ = None
slot_manager = None
tray_manager = None
vision_cli = None
gripper = None
gripper_manager = None
T_GRIPPER_TO_CAM = None

# ========================================
# 그리퍼 드라이버
# ========================================
try:
    from ff_robot.onrobot import RG
    print("✅ Real Gripper Driver Loaded")
except ImportError:
    class RG:
        def __init__(self, *args): pass
        def open_gripper(self): print("   👐 [Virtual] Open")
        def close_gripper(self, force=None): print("   ✊ [Virtual] Close")
        def move_gripper(self, width): print(f"   👌 [Virtual] Move Width: {width}")

# ========================================
# 유틸리티 함수
# ========================================
def load_calibration():
    """캘리브레이션 파일 로드 및 보정 적용"""
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    
    if os.path.exists(npy_path):
        T_base = np.load(npy_path)
        T_base[0, 3] += (CALIB_FIX_X / 1000.0)
        T_base[1, 3] += (CALIB_FIX_Y / 1000.0)
        T_base[2, 3] += (CALIB_FIX_Z / 1000.0)
        T_GRIPPER_TO_CAM = T_base
        print(f"✅ 캘리브레이션 로드: {npy_path}")
    else:
        print(f"❌ 캘리브레이션 파일 없음!")
        sys.exit(1)

def transform_camera_to_base(cam_xyz):
    """카메라 좌표 → 로봇 베이스 좌표 변환"""
    from DSR_ROBOT2 import get_current_posx
    from scipy.spatial.transform import Rotation as R
    
    try:
        curr_posx = get_current_posx()
        if curr_posx is None: return None
        if isinstance(curr_posx, tuple): curr_posx = curr_posx[0]
        
        # 로봇 현재 자세 행렬
        x, y, z, rx, ry, rz = curr_posx
        rot = R.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T_base_gripper = np.eye(4)
        T_base_gripper[:3, :3] = rot
        T_base_gripper[:3, 3] = [x, y, z]
        
        # 베이스 → 카메라
        T_base_cam = T_base_gripper @ T_GRIPPER_TO_CAM
        
        # 카메라 좌표 변환
        p_cam = np.array([cam_xyz[0]*1000, cam_xyz[1]*1000, cam_xyz[2]*1000, 1.0])
        p_base = T_base_cam @ p_cam
        return p_base[:3]
    except Exception as e:
        print(f"   ❌ 변환 에러: {e}")
        return None

def wait_for_motion():
    """모션 완료 대기"""
    from DSR_ROBOT2 import check_motion
    time.sleep(0.1)
    while check_motion() != 0:
        time.sleep(0.05)
        if not rclpy.ok(): return False
    return True

def safe_movej(joints, vel=40.0, acc=40.0, desc="이동"):
    """Joint 이동"""
    global node_
    from DSR_ROBOT2 import movej
    try:
        time.sleep(0.05)
        movej(joints, vel=vel, acc=acc)
        wait_for_motion()
        time.sleep(0.05)
        return True
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

def safe_movel(pos, vel=100.0, acc=100.0, desc="이동"):
    """Linear 이동"""
    global node_
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    try:
        time.sleep(0.05)
        movel(pos, vel=[vel, vel], acc=[acc, acc], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05)
        return True
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

def call_vision_service(target_name):
    """비전 서비스 호출"""
    global node_, vision_cli
    
    if vision_cli is None or not vision_cli.service_is_ready():
        return None
    
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 10.0:
            return None
        time.sleep(0.1)
    
    try:
        res = future.result()
        if res.found:
            return {
                "position": [res.position.x, res.position.y, res.position.z],
                "rotation": [res.rx, res.ry, res.rz],
                "confidence": res.confidence
            }
    except:
        pass
    
    return None

# ========================================
# [Phase 1] 트레이 픽업
# ========================================
def pick_and_place_tray(target_slot_id):
    """
    트레이를 인식하여 작업 테이블에 배치
    
    Returns:
        success: bool
    """
    global node_, tray_manager, gripper
    
    node_.get_logger().info(f"========================================")
    node_.get_logger().info(f"[Phase 1] 트레이 픽업 → 슬롯 {target_slot_id}")
    node_.get_logger().info(f"========================================")
    
    # 1. 트레이 관측 위치로 이동
    node_.get_logger().info("🔭 트레이 관측 위치 이동...")
    if not safe_movej(J_TRAY_OBSERVE):
        return False
    
    time.sleep(1.0)  # 안정화 대기
    
    # 2. 트레이 인식 (2개 모두 찾기)
    node_.get_logger().info("🔎 트레이 인식 중...")
    
    # [TODO] 실제로는 여러 개 찾을 수 있도록 수정 필요
    # 현재는 1개씩만 찾음
    vision_result = call_vision_service("tray")
    
    if vision_result is None:
        node_.get_logger().error("❌ 트레이를 찾을 수 없습니다!")
        return False
    
    # 3. 카메라 좌표 → 베이스 좌표 변환
    cam_pos = vision_result["position"]
    base_pos = transform_camera_to_base(cam_pos)
    
    if base_pos is None:
        return False
    
    tray_rotation_rz = vision_result["rotation"][2]  # Yaw만 사용
    
    node_.get_logger().info(f"   📍 트레이 위치: {base_pos}")
    node_.get_logger().info(f"   🔄 트레이 회전: {tray_rotation_rz:.1f}°")
    
    # 4. 끄트머리 그립 포인트 계산
    grip_point = tray_manager.calculate_tray_grip_point(base_pos, tray_rotation_rz)
    
    node_.get_logger().info(f"   🎯 그립 포인트: {grip_point}")
    
    # 5. 접근 → 그립 → 상승
    approach_point = grip_point.copy()
    approach_point[2] += APPROACH_HEIGHT
    
    if gripper:
        gripper.open_gripper()
    
    if not safe_movel(approach_point, desc="트레이 접근"):
        return False
    
    if not safe_movel(grip_point, desc="트레이 하강"):
        return False
    
    if gripper:
        gripper.close_gripper()
        time.sleep(0.5)
    
    if not safe_movel(approach_point, desc="트레이 상승"):
        return False
    
    # 6. 작업 테이블로 이동
    work_pos = tray_manager.calculate_work_table_position(target_slot_id)
    
    node_.get_logger().info(f"🚚 작업 테이블로 이동: 슬롯 {target_slot_id}")
    
    work_approach = work_pos.copy()
    work_approach[2] += APPROACH_HEIGHT
    
    if not safe_movel(work_approach, desc="작업 테이블 접근"):
        return False
    
    if not safe_movel(work_pos, desc="트레이 배치"):
        return False
    
    if gripper:
        gripper.open_gripper()
        time.sleep(0.3)
    
    if not safe_movel(work_approach, desc="상승"):
        return False
    
    # 7. 트레이 상태 업데이트
    tray_manager.update_tray_status(target_slot_id, "placed")
    
    node_.get_logger().info(f"✅ 트레이 {target_slot_id} 배치 완료!")
    
    return True

# ========================================
# [Phase 2] 물품 배치
# ========================================
def pick_and_place_item(item_name, target_slot_id):
    """
    물품을 인식하여 트레이에 배치
    
    Args:
        item_name: 물품 이름
        target_slot_id: 배치할 트레이 슬롯 (0 또는 1)
    
    Returns:
        success: bool
    """
    global node_, gripper_manager, tray_manager
    
    node_.get_logger().info(f"🔎 [Search] '{item_name}' → 슬롯 {target_slot_id}")
    
    # 1. 물품 관측 위치로 이동
    if not safe_movej(J_ITEM_OBSERVE):
        return False
    
    time.sleep(1.0)
    
    # 2. 물품 인식
    vision_result = call_vision_service(item_name)
    
    if vision_result is None:
        node_.get_logger().warn(f"   ⚠️ '{item_name}' 못 찾음")
        return False
    
    # 3. 좌표 변환
    cam_pos = vision_result["position"]
    base_pos = transform_camera_to_base(cam_pos)
    
    if base_pos is None:
        return False
    
    # 4. 오프셋 적용
    specific_offset = ITEM_OFFSETS.get(item_name, {"x":0.0, "y":0.0, "z":0.0})
    
    target_x = base_pos[0] + GLOBAL_OFFSET_X + specific_offset.get("x", 0.0)
    target_y = base_pos[1] + GLOBAL_OFFSET_Y + specific_offset.get("y", 0.0)
    target_z = base_pos[2] + GLOBAL_OFFSET_Z + specific_offset.get("z", 0.0)
    
    target_rx = specific_offset.get("rx", 0.0)
    target_ry = specific_offset.get("ry", 180.0)
    target_rz = specific_offset.get("rz", 0.0)
    
    if target_z < SAFE_Z_FLOOR_LIMIT:
        target_z = SAFE_Z_FLOOR_LIMIT
    
    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    approach_pose = [target_x, target_y, target_z + APPROACH_HEIGHT, target_rx, target_ry, target_rz]
    
    # 5. 픽업
    if not safe_movel(approach_pose, desc="물품 접근"):
        return False
    
    if gripper_manager:
        gripper_manager.prepare_grip(item_name)
    
    if not safe_movel(pick_pose, desc="물품 하강"):
        return False
    
    if gripper_manager:
        gripper_manager.execute_grip(item_name)
    else:
        if gripper:
            gripper.close_gripper()
        time.sleep(0.5)
    
    if not safe_movel(approach_pose, desc="물품 상승"):
        return False
    
    # 6. 트레이 위치 계산 (Order Logic 사용)
    # [TODO] slot_manager와 통합 필요
    # 현재는 간단하게 트레이 중앙에 배치
    tray_pos = tray_manager.calculate_work_table_position(target_slot_id)
    
    place_pos = tray_pos.copy()
    place_approach = place_pos.copy()
    place_approach[2] += APPROACH_HEIGHT
    
    # 7. 배치
    node_.get_logger().info(f"📦 트레이 {target_slot_id}에 배치...")
    
    if not safe_movel(place_approach, desc="트레이 접근"):
        return False
    
    if not safe_movel(place_pos, desc="물품 배치"):
        return False
    
    if gripper_manager:
        gripper_manager.release(item_name)
    else:
        if gripper:
            gripper.open_gripper()
        time.sleep(0.3)
    
    if not safe_movel(place_approach, desc="상승"):
        return False
    
    node_.get_logger().info(f"✅ '{item_name}' 배치 완료!")
    
    return True

# ========================================
# [Phase 3] 트레이 서빙
# ========================================
def serve_tray(slot_id):
    """
    완료된 트레이를 서빙 테이블로 이동
    
    Returns:
        success: bool
    """
    global node_, tray_manager, gripper
    
    node_.get_logger().info(f"========================================")
    node_.get_logger().info(f"[Phase 3] 트레이 {slot_id} 서빙")
    node_.get_logger().info(f"========================================")
    
    # 1. 작업 테이블의 트레이 위치
    work_pos = tray_manager.calculate_work_table_position(slot_id)
    
    # 트레이 끄트머리 (Y축 양의 방향)
    # [수정 필요] 트레이 크기에 맞게 조정
    grip_offset_y = 100.0  # mm
    
    grip_pos = work_pos.copy()
    grip_pos[1] += grip_offset_y  # Y축으로 이동
    
    grip_approach = grip_pos.copy()
    grip_approach[2] += APPROACH_HEIGHT
    
    # 2. 그립
    if gripper:
        gripper.open_gripper()
    
    if not safe_movel(grip_approach, desc="트레이 접근"):
        return False
    
    if not safe_movel(grip_pos, desc="트레이 그립"):
        return False
    
    if gripper:
        gripper.close_gripper()
        time.sleep(0.5)
    
    if not safe_movel(grip_approach, desc="상승"):
        return False
    
    # 3. 서빙 테이블로 이동
    serve_pos = tray_manager.calculate_serve_position()
    
    node_.get_logger().info("🍽️ 서빙 테이블로 이동...")
    
    serve_approach = serve_pos.copy()
    serve_approach[2] += APPROACH_HEIGHT
    
    if not safe_movel(serve_approach, desc="서빙 테이블 접근"):
        return False
    
    if not safe_movel(serve_pos, desc="트레이 서빙"):
        return False
    
    if gripper:
        gripper.open_gripper()
        time.sleep(0.3)
    
    if not safe_movel(serve_approach, desc="상승"):
        return False
    
    # 4. 트레이 상태 업데이트
    tray_manager.update_tray_status(slot_id, "empty")
    
    node_.get_logger().info(f"🎉 트레이 {slot_id} 서빙 완료!")
    
    return True

# ========================================
# 메인 태스크 루프
# ========================================
def perform_robot_task():
    """메인 워크플로우"""
    global node_, slot_manager, tray_manager, gripper, gripper_manager
    
    try:
        from DSR_ROBOT2 import movej
    except:
        return
    
    # 그리퍼 연결
    try:
        gripper = RG("rg2", "192.168.1.1", "502")
        node_.get_logger().info("✅ Real Gripper Connected")
    except:
        gripper = RG()
    
    gripper_manager = GripperManager(gripper)
    
    node_.get_logger().info("[Task] 🤖 로봇 준비 완료.")
    
    while rclpy.ok():
        try:
            # [단계 1] 빈 슬롯이 있고, 주문이 있으면 트레이 가져오기
            empty_slot = tray_manager.get_empty_slot()
            
            if empty_slot is not None and slot_manager.get_all_needed_items():
                node_.get_logger().info(f"📥 새 트레이 필요 (슬롯 {empty_slot})")
                
                if pick_and_place_tray(empty_slot):
                    tray_manager.update_tray_status(empty_slot, "working")
            
            # [단계 2] 물품 배치
            needed_items = slot_manager.get_all_needed_items()
            
            if needed_items:
                item = needed_items[0]
                
                # [TODO] Order Logic과 통합하여 어느 슬롯에 배치할지 결정
                # 현재는 간단하게 working 상태인 슬롯 찾기
                target_slot = None
                for sid, state in tray_manager.tray_states.items():
                    if state["status"] == "working":
                        target_slot = sid
                        break
                
                if target_slot is not None:
                    if pick_and_place_item(item, target_slot):
                        # [TODO] slot_manager 업데이트
                        pass
            
            # [단계 3] 완료된 트레이 서빙
            # [TODO] Order Logic과 통합하여 완료 여부 판단
            
            time.sleep(1.0)
            
        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

# ========================================
# 주문 서비스 핸들러
# ========================================
def handle_order_request(request, response):
    """키오스크 주문 처리"""
    global slot_manager, node_
    
    node_.get_logger().info(f"⚡ [Service] 주문: {request.item_names}")
    
    success, msg = slot_manager.check_and_deduct_stock(
        request.item_names, 
        request.item_quantities
    )
    
    if success:
        order_id = f"ORD-{int(time.time())}"
        slot_manager.add_order_to_slot(
            order_id, 
            request.item_names, 
            request.item_quantities
        )
        response.assigned_order_id = order_id
        response.message = "접수 완료"
        node_.get_logger().info(f"✅ 주문 {order_id} 처리됨")
    else:
        response.assigned_order_id = ""
        response.message = msg
        node_.get_logger().warn(f"🚫 주문 거절: {msg}")
    
    # 재고 현황 반환
    item_ids, item_counts = slot_manager.get_inventory_status()
    response.all_item_ids = item_ids
    response.all_item_counts = item_counts
    
    return response

# ========================================
# 메인 함수
# ========================================
def main(args=None):
    global node_, dsr_control_node_, slot_manager, tray_manager, vision_cli
    
    rclpy.init(args=args)
    
    # 초기화
    load_calibration()
    slot_manager = SlotManager()
    tray_manager = TrayManager()
    
    # 노드 생성
    node_ = rclpy.create_node("robot_controller_node", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_
    
    # 서비스 및 클라이언트
    cb_group = ReentrantCallbackGroup()
    node_.create_service(OrderService, '/dsr01/order_service', 
                        handle_order_request, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', 
                                     callback_group=cb_group)
    
    # Executor
    executor = MultiThreadedExecutor()
    executor.add_node(node_)
    
    # 로봇 태스크 스레드
    t_robot = threading.Thread(target=perform_robot_task, daemon=True)
    t_robot.start()
    
    try:
        executor.spin()
    except:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
