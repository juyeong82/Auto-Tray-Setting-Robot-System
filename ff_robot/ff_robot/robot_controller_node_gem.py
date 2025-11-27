#!/usr/bin/env python3
# robot_controller_node_gem.py (Robust Version)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import DR_init
from ff_robot_interfaces.srv import OrderService, DetectObject
from ff_robot.order_logic import SlotManager

# --- 가상 그리퍼 클래스 ---
class RG:
    def __init__(self, *args): 
        print("[Gripper] 가상 그리퍼 초기화 완료")
    def open_gripper(self): 
        print("   👐 [Virtual] 그리퍼 열림 (Open)")
    def close_gripper(self): 
        print("   ✊ [Virtual] 그리퍼 닫힘 (Close)")
    def get_status(self): 
        return "Virtual OK"

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# --- 전역 변수 ---
node_ = None          
dsr_control_node_ = None # 로봇 제어용 비밀 노드
manager = None
vision_cli = None 
gripper = None

# ==============================================================================
# [Helper] Vision Service Call
# ==============================================================================
def call_vision_service(target_name):
    global node_, vision_cli
    if vision_cli is None or not vision_cli.service_is_ready():
        return None
    
    req = DetectObject.Request()
    req.target_object_id = target_name
    
    future = vision_cli.call_async(req)
    
    timeout = 3.0
    start_time = time.time()
    
    while not future.done():
        if time.time() - start_time > timeout:
            node_.get_logger().warn(f"Vision 타임아웃: {target_name}")
            return None
        time.sleep(0.01)
        
    try:
        res = future.result()
        if res.found:
            p = res.position
            q = res.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            rx, ry, rz = rot.as_euler('xyz', degrees=True)
            return [p.x, p.y, p.z, rx, ry, rz]
    except Exception as e:
        node_.get_logger().error(f"Vision 호출 에러: {e}")
    return None

# ==============================================================================
# [Helper] Robot Motion Functions (타이밍 강화)
# ==============================================================================
def safe_movel(pos, desc="이동"):
    global node_
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    
    node_.get_logger().info(f"   🏃 {desc}... {pos[:3]}")
    
    for i in range(3):
        try:
            # [수정] 대기 시간을 1.0초 -> 1.5초로 늘림 (안전 확보)
            time.sleep(1.0) 
            
            # 이동 명령
            movel(pos, vel=[100, 100], acc=[100, 100], ref=DR_BASE, mod=DR_MV_MOD_ABS)
            
            # [수정] 명령 후 대기 시간도 0.5초 -> 1.0초로 늘림
            time.sleep(1.0)
            
            return True 
        except Exception as e:
            node_.get_logger().warn(f"   ⚠️ {desc} 실패 ({i+1}/3): {e}")
            time.sleep(1.0)
        # [추가] 시스템 에러 등 치명적 에러도 잡기 위해 BaseException 추가
        except BaseException as e:
            node_.get_logger().error(f"   ❌ {desc} 치명적 오류 발생: {e}")
            time.sleep(1.0)
            
    node_.get_logger().error(f"   ❌ {desc} 최종 실패")
    return False

def safe_move_and_pick(target_pos):
    global node_, gripper
    
    target_pos = list(target_pos[:3]) + [0, 180, 0]
    # 접근 위치 (Z + 100mm)
    approach_pos = list(target_pos) 
    approach_pos[2] += 100.0        

    # 1. 접근
    if not safe_movel(approach_pos, "접근 중"): return False
    
    if gripper: 
        gripper.open_gripper()
        time.sleep(0.5)

    # 2. 하강
    if not safe_movel(target_pos, "하강 및 파지"): return False
    
    if gripper:
        gripper.close_gripper()
        time.sleep(0.5)

    # 3. 상승 (여기서 에러가 났었음 -> 이제 예외 처리됨)
    if not safe_movel(approach_pos, "들어올리기"): return False
    
    return True

def safe_move_and_place(target_pos):
    global node_, gripper

    target_pos = list(target_pos[:3]) + [0, 180, 0]

    approach_pos = list(target_pos)
    approach_pos[2] += 100.0

    if not safe_movel(approach_pos, "배치 위치 접근"): return False
    if not safe_movel(target_pos, "배치 위치 하강"): return False

    if gripper:
        gripper.open_gripper()
        time.sleep(0.5)

    if not safe_movel(approach_pos, "배치 후 복귀"): return False
    
    return True

# ==============================================================================
# [Task] Robot Workflow Thread
# ==============================================================================
def perform_robot_task():
    global node_, manager, gripper
    
    # (초기화 코드는 동일...)
    try:
        from DSR_ROBOT2 import movej, posj
    except ImportError:
        node_.get_logger().error("DSR_ROBOT2 로드 실패.")
        return

    gripper = RG() 
    node_.get_logger().info("✅ 가상 그리퍼 모드 활성화")
    
    try:
        home_pos = posj(0, 0, 90, 0, 90, 0)
    except: pass
    
    node_.get_logger().info("[Task] 로봇 준비 완료. 주문 대기 중...")

    while rclpy.ok():
        try:
            needed_items = manager.get_all_needed_items()
            
            if not needed_items:
                time.sleep(1.0)
                continue

            target_item = needed_items[0]
            node_.get_logger().info(f"🔎 [Search] '{target_item}' 찾는 중...")

            item_pose = call_vision_service(target_item)
            if item_pose is None:
                time.sleep(1.0)
                continue

            tray0_pose = call_vision_service("tray_marker_0")
            tray1_pose = call_vision_service("tray_marker_1")
            
            if not tray0_pose: 
                tray0_pose = [400, 200, 0, 0, 180, 0]
                tray1_pose = [400, -200, 0, 0, 180, 0]

            final_pos, action = manager.process_item_placement(
                target_item, tray0_pose, tray1_pose
            )

            if final_pos:
                # [핵심 수정] 좌표(3개) 뒤에 자세(3개)를 붙여서 6개를 만듦
                # [0, 180, 0]은 그리퍼가 바닥을 보는 표준 자세
                if len(final_pos) == 3:
                    final_pos.extend([0.0, 180.0, 0.0])
                
                node_.get_logger().info(f"🚀 [Action] '{action}' 시작")
                
                try:
                    # 1. Pick 시도
                    if not safe_move_and_pick(item_pose):
                        node_.get_logger().warn("Pick 실패 -> Skip")
                        continue 

                    # ========================================================
                    # [핵심 수정] Pick과 Place 사이에 3초간 확실한 휴식 부여
                    # ========================================================
                    node_.get_logger().info("   ⏳ 이동 준비 중 (3초 대기)...")
                    time.sleep(3.0) 

                    # 2. Place 시도
                    if not safe_move_and_place(final_pos):
                        node_.get_logger().warn("Place 실패 -> Skip")
                        continue

                    # 3. 완료 처리
                    if "SWAP" in action or "FINISH" in action:
                        node_.get_logger().info("   🚚 [Serving] 서빙 완료")
                        if action == "PLACE_AND_FINISH":
                            manager.clear_slot(0) 
                            manager.clear_slot(1) 

                    node_.get_logger().info("✅ 사이클 완료\n")
                    
                    # 다음 아이템 잡으러 가기 전에도 휴식
                    time.sleep(2.0)
                    
                except Exception as e:
                    node_.get_logger().error(f"동작 중 오류: {e}")
                    time.sleep(2.0)

        except Exception as e:
            node_.get_logger().error(f"Task Loop Error: {e}")
            time.sleep(1.0)

# ==============================================================================
# [Service] Order Callback
# ==============================================================================
def handle_order_request(request, response):
    global manager, node_
    
    node_.get_logger().info(f"⚡ [Service] 주문 요청 수신: {request.item_names}")
    
    item_names = request.item_names
    item_quantities = request.item_quantities
    
    if len(item_names) != len(item_quantities):
        response.message = "데이터 길이 오류"
        return response
        
    success, msg = manager.check_and_deduct_stock(item_names, item_quantities)
    
    if success:
        new_order_id = f"ORD-{int(time.time())}"
        manager.add_order_to_slot(new_order_id, item_names, item_quantities)
        response.assigned_order_id = new_order_id
        response.message = "주문 접수 완료"
        response.estimated_time = sum(item_quantities) * 30
        node_.get_logger().info(f"✅ 주문 접수 완료")
    else:
        response.assigned_order_id = ""
        response.message = msg
        
    ids, counts = manager.get_inventory_status()
    response.all_item_ids = ids
    response.all_item_counts = counts
    return response

# ==============================================================================
# [Main] 
# ==============================================================================
def main(args=None):
    global node_, dsr_control_node_, manager, vision_cli
    
    rclpy.init(args=args)
    manager = SlotManager()
    
    # 노드 2개 생성 (통신용, 제어용)
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
    
    try:
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