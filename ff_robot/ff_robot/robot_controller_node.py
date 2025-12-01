#!/usr/bin/env python3
# robot_controller_node.py (Threaded Fix + GripperManager + 2-step Pick)
# MISSION 
# 음성 매끄럽게!!!
# 키오스크 주문 리스트 수량 확인!!!
# # - 수량 데이터(item_quantities)를 순수 list로 변환하여 전송 (array 문제 해결)  -  누가 지웠던데 지우지 마세요 ㅜㅜ


import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading  # [핵심] 스레딩 모듈 복구
import time
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

import DR_init
from ff_robot_interfaces.srv import OrderService, DetectObject
from ff_robot.order_logic import SlotManager
from ff_robot.gripper import GripperManager   # ✅ 추가: 메뉴별 그리퍼 제어

# [설정] 로그 즉시 출력
sys.stdout.reconfigure(line_buffering=True)

# --- [설정] 안전 검증 높이 ---
SAFE_Z_HEIGHT = 400.0 

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# [설정] 관측 자세
J_LOOK_POS = [43.97, -5.17, 78.33, -43.15, 126.28, -154.88]

# [설정] 임시 Place 위치
TEMP_PLACE_POS = [300.0, 10.0, 400.0, 0.0, 180.0, 0.0]

# --- 전역 변수 ---
node_ = None          
dsr_control_node_ = None 
manager = None
vision_cli = None 
gripper = None
gripper_manager = None   # ✅ 추가
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
    # DSR Node 사용
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
# [Motion]
# ==============================================================================
def wait_for_motion():
    from DSR_ROBOT2 import check_motion
    # [수정] 체크 전 대기 시간 단축
    # time.sleep(0.2) -> 0.1
    time.sleep(0.1)
    while check_motion() != 0:
        time.sleep(0.05) # 루프 대기 시간 단축
        if not rclpy.ok(): return False
    return True

def safe_movel(pos, desc="이동"):
    global node_
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    
    # 로그도 너무 많이 찍으면 느려지니 필요할 때만
    # node_.get_logger().info(f"   🏃 {desc}...") 
    
    try:
        # [수정 1] 시작 전 대기: 1.0초 -> 0.05초 (거의 삭제)
        # time.sleep(1.0) 
        time.sleep(0.05)
        
        # [수정 2] 속도/가속도 증가: 40 -> 100 (원래 속도로 복구)
        movel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        
        wait_for_motion()
        
        # [수정 3] 종료 후 대기: 0.5초 -> 0.05초
        # time.sleep(0.5) 
        time.sleep(0.05)
        
        return True 
    except Exception as e:
        node_.get_logger().error(f"   ❌ {desc} 오류: {e}")
        return False

def safe_move_and_check(target_pos):
    """
    target_pos: transform_camera_to_base() 결과 [x, y, z] (mm 단위)

    1) calibration 보정 오프셋 적용
       - z += 170
       - y -= 20
    2) 보정된 픽 포인트 기준:
       - (x, y, z+50) 상공으로 이동
       - (x, y, z)로 50mm 직하 이동
    """
    global node_
    
    # --- 기존 오프셋 유지 ---
    safe_target = list(target_pos[:3])
    safe_target[2] += 170.0   # z 보정
    safe_target[1] -= 20.0    # y 보정
    
    x, y, z = safe_target

    # 1. 타겟 상공 (z + 50mm)
    approach_pose = [float(x), float(y), float(z + 50.0), 0.0, 180.0, 0.0]
    node_.get_logger().info(f"   🛡️ 상공 접근: X={x:.1f}, Y={y:.1f}, Z={z+50.0:.1f}mm")
    if not safe_movel(approach_pose, "Pick 상공 이동"):
        return False

    # 2. 타겟 위치까지 직하 이동 (보정된 z)
    pick_pose = [float(x), float(y), float(z), 0.0, 180.0, 0.0]
    node_.get_logger().info(f"   🎯 픽 위치 이동: X={x:.1f}, Y={y:.1f}, Z={z:.1f}mm")
    if not safe_movel(pick_pose, "Pick 위치 이동"):
        return False

    return True

# ==============================================================================
# [Vision Service]
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
        time.sleep(0.1) # 스레드 양보
        
    try:
        res = future.result()
        if res.found: return [res.position.x, res.position.y, res.position.z]
    except: pass
    return None

# ==============================================================================
# [Task] 로봇 행동 스레드
# ==============================================================================
def perform_robot_task():
    global node_, manager, gripper, gripper_manager
    
    # DSR 라이브러리 임포트 (스레드 내부)
    try: 
        from DSR_ROBOT2 import movej
    except: 
        return

    # --- 그리퍼 초기화 ---
    try:
        gripper = RG("rg2", "192.168.1.1", "502") 
        node_.get_logger().info("✅ Real Gripper Connected")
    except Exception as e:
        node_.get_logger().warn(f"⚠️ Real Gripper 연결 실패, Virtual Gripper 사용: {e}")
        gripper = RG()

    # --- GripperManager 생성 ---
    try:
        gripper_manager = GripperManager(gripper)
        node_.get_logger().info("✅ GripperManager 초기화 완료")
    except Exception as e:
        gripper_manager = None
        node_.get_logger().error(f"❌ GripperManager 초기화 실패: {e}")

    node_.get_logger().info(f"🔭 관측 위치 이동... {J_LOOK_POS}")
    try:
        movej(J_LOOK_POS, vel=40.0, acc=40.0)
        wait_for_motion()
    except Exception as e:
        node_.get_logger().error(f"❌ 초기 이동 실패: {e}")
    
    node_.get_logger().info("[Task] 준비 완료. 루프 시작.")

    while rclpy.ok():
        try:
            needed_items = manager.get_all_needed_items()
            if not needed_items:
                time.sleep(1.0) # 할 일 없으면 대기
                continue

            target_item = needed_items[0]
            node_.get_logger().info(f"========================================")
            node_.get_logger().info(f"🔎 [Search] '{target_item}'")

            # 1. 관측 위치 복귀
            try:
                movej(J_LOOK_POS, vel=40.0, acc=40.0)
                wait_for_motion()
            except: 
                pass

            # 1-1. 관측 자세에서 메뉴에 맞게 그리퍼 오픈
            if gripper_manager:
                gripper_manager.prepare_grip(target_item)
            else:
                node_.get_logger().warn("   ⚠️ GripperManager 없음 -> 그리퍼 준비 생략")

            # 2. 비전 호출
            cam_xyz = call_vision_service(target_item)
            if cam_xyz is None:
                node_.get_logger().warn("   ⚠️ 타겟 못 찾음 (Retry)")
                time.sleep(1.0)
                continue

            # 3. 변환
            base_xyz = transform_camera_to_base(cam_xyz)
            if base_xyz is None:
                time.sleep(1.0)
                continue

            # 4. 상공 → 픽 위치까지 이동
            if safe_move_and_check(base_xyz):
                node_.get_logger().info("✅ 위치 접근 완료. 집기 시도.")

                # 4-1. 현재 위치에서 그리퍼 클로즈
                if gripper_manager:
                    gripper_manager.execute_grip(target_item)
                else:
                    node_.get_logger().warn("   ⚠️ GripperManager 없음 -> 집기 생략")
                time.sleep(0.5)

                # 4-2. Place 위치로 이동 후 release
                if safe_movel(TEMP_PLACE_POS, "Place"):
                    if gripper_manager:
                        gripper_manager.release(target_item)
                    else:
                        node_.get_logger().warn("   ⚠️ GripperManager 없음 -> 릴리즈 생략")

                    manager.clear_slot(0) 
                    node_.get_logger().info("🎉 작업 완료.")
            else:
                node_.get_logger().error("❌ 이동 실패")

            # [수정] 작업 사이 대기 시간 단축
            # time.sleep(1.0) -> 0.1
            time.sleep(0.1)

        except Exception as e:
            node_.get_logger().error(f"Task Error: {e}")
            time.sleep(1.0)

# ==============================================================================
# [Main] 서비스 핸들러 및 실행기
# ==============================================================================
def handle_order_request(request, response):
    global manager, node_
    node_.get_logger().info(f"⚡ [Service] 주문: {request.item_names}, {list(request.item_quantities)}")
    success, msg = manager.check_and_deduct_stock(request.item_names, request.item_quantities)
    
    if success:
        manager.add_order_to_slot(f"ORD-{int(time.time())}", request.item_names, request.item_quantities)
        
        # [수정] response.success = True  <-- 이 줄 삭제!!! (srv 파일에 success 필드가 없음)
        
        # 대신 성공 여부는 assigned_order_id가 채워진 것으로 판단하거나, message로 전달
        response.assigned_order_id = f"ORD-{int(time.time())}" # ID 채우기
        response.message = "접수 완료"
        
        node_.get_logger().info("✅ 주문 처리됨 -> Task 스레드가 감지할 것임")
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
    
    # 1. 통신 노드 (Spin용)
    node_ = rclpy.create_node("robot_controller_node", namespace=ROBOT_ID)
    
    # 2. 로봇 제어 노드 (DSR용 - Spin 안함)
    dsr_control_node_ = rclpy.create_node("dsr_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_ 

    cb_group = ReentrantCallbackGroup()
    node_.create_service(OrderService, '/dsr01/order_service', handle_order_request, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)

    executor = MultiThreadedExecutor()
    executor.add_node(node_) # 통신 노드만 등록

    # [핵심 수정] 별도 스레드에서 로봇 로직 실행
    t_robot = threading.Thread(target=perform_robot_task, daemon=True)
    t_robot.start()
    
    try:
        # 메인 스레드는 ROS 통신에 집중 (주문 받기, 비전 요청/응답 처리)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            executor.shutdown()
            node_.destroy_node()
            dsr_control_node_.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()

