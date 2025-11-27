#!/usr/bin/env python3
# robot_controller_node.py
#
# 햄버거집 메인 로봇 컨트롤러 (두산 로봇 스크립트 방식)
# - 키오스크로부터 주문 수신 (Service Server)
# - 재고 관리 및 로봇 작업 스케줄링
# - 비전 노드로부터 물체 위치 수신 (Subscriber)

import rclpy
from rclpy.node import Node
import threading
import time
import queue

# --- 두산 로봇 관련 임포트 (가상 환경 또는 실제 로봇) ---
import DR_init


# --- 인터페이스 임포트 ---
from ff_robot_interfaces.srv import OrderService
from ff_robot_interfaces.msg import DetectedObject
from std_msgs.msg import String

# --- 로봇 설정 상수 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# --- 전역 변수 ---
node_ = None
order_queue = queue.PriorityQueue() # 주문 대기열 (우선순위 큐)
inventory = {                       # 초기 재고 (예시)
    "burger_cheese": 5,
    "burger_bulgogi": 5,
    "fries": 10,
    "coke": 10
}
current_vision_data = None          # 가장 최근 감지된 물체

# ==============================================================================
# [1] 로봇 제어 스레드 (Task Thread)
# ==============================================================================
def perform_robot_task():
    global node_, order_queue, inventory, current_vision_data
    
    try:
        from DSR_ROBOT2 import (
            set_tool, set_tcp, amovel, movel, wait, check_motion,
            DR_BASE, DR_MV_MOD_REL, DR_MV_MOD_ABS
        )
    except ImportError as e:
        node_.get_logger().error(f"DSR_ROBOT2 임포트 실패: {e}")
        return
    
    # 로봇 초기화 (Tool, TCP 설정 등)
    try:
        # 실제 로봇 연결 시 활성화
        # set_tool("Gripper_V1")
        # set_tcp("TCP_V1")
        pass
    except Exception as e:
        if node_: node_.get_logger().error(f"로봇 초기화 오류 (시뮬레이션이면 무시): {e}")

    node_.get_logger().info("[Robot Task] 로봇 제어 루프 시작. 주문 대기 중...")

    while rclpy.ok():
        try:
            # 1. 대기열에서 주문 확인 (Blocking 아님)
            if order_queue.empty():
                time.sleep(1.0)
                continue

            # 2. 주문 꺼내기 (우선순위, 주문데이터)
            priority, order_data = order_queue.get()
            order_id = order_data['order_id']
            items = order_data['items']
            
            node_.get_logger().info(f"🚀 [작업 시작] 주문번호: {order_id} / 우선순위: {priority}")

            # 3. 주문 아이템별로 작업 수행
            for item_name, qty in items.items():
                for i in range(qty):
                    node_.get_logger().info(f"   -> '{item_name}' ({i+1}/{qty}) 조리/서빙 준비 중...")
                    
                    # (A) 카메라 구도 전환 (가정)
                    # move_to_camera_view_pose() 
                    
                    # (B) 비전 데이터 대기 (비전 노드가 있다고 가정)
                    # 실제 구현 시: 원하는 물체가 감지될 때까지 wait
                    target_pos = None
                    if current_vision_data and current_vision_data.object_id == item_name:
                        target_pos = current_vision_data.position
                        node_.get_logger().info(f"   📸 물체 감지됨: {item_name} at {target_pos}")
                    else:
                        node_.get_logger().warn(f"   ⚠️ 물체 감지 실패 (Mock Data 사용)")
                        # 예외처리: 감지 안되면 기본 위치 사용 등

                    # (C) 로봇 이동 (Pick & Place)
                    # move_to_pick(target_pos)
                    # gripper_close()
                    # move_to_tray()
                    # gripper_open()
                    
                    # --- [Mock] 로봇 동작 시뮬레이션 ---
                    time.sleep(2.0) # 이동 시간 시뮬레이션
                    node_.get_logger().info(f"   ✅ '{item_name}' 트레이 배치 완료")

            node_.get_logger().info(f"🏁 [작업 완료] 주문번호: {order_id} 서빙 완료\n")
            
        except Exception as e:
            node_.get_logger().error(f"작업 스레드 예외 발생: {e}")
            time.sleep(1.0)


# ==============================================================================
# [2] ROS2 통신 스레드 (Service & Subscriber)
# ==============================================================================

# 2-1. 주문 서비스 콜백
def handle_order_request(request, response):
    global inventory, order_queue
    
    node_.get_logger().info(f"📩 주문 수신: {request.item_names} (수량: {request.item_quantities})")

    # 1. 재고 확인
    insufficient_items = []
    temp_inventory = inventory.copy() # 가검증용 복사본
    
    for name, qty in zip(request.item_names, request.item_quantities):
        if name not in temp_inventory:
            response.message = f"존재하지 않는 메뉴: {name}"
            response.assigned_order_id = ""
            return response
        if temp_inventory[name] < qty:
            insufficient_items.append(name)
        else:
            temp_inventory[name] -= qty

    # 2. 재고 부족 시 거절 응답
    if insufficient_items:
        response.message = f"재고 부족: {', '.join(insufficient_items)}"
        response.assigned_order_id = "" # 실패 표시
        response.estimated_time = 0
        # 현재 재고 상태 반환
        response.all_item_ids = list(inventory.keys())
        response.all_item_counts = list(inventory.values())
        node_.get_logger().warn(f"❌ 주문 거절: {response.message}")
        return response

    # 3. 주문 승인 및 재고 차감 (실제 반영)
    inventory = temp_inventory
    order_id = f"ORD-{int(time.time())}" # 간단한 ID 생성
    
    # 4. 작업 대기열 등록 (Priority Queue)
    # 로직: 아이템 수가 적을수록 빨리 처리 (Simple Logic) 
    # 혹은 VIP 등의 조건 추가 가능. 여기선 기본 10, 아이템 수 많으면 후순위
    priority = 10 + sum(request.item_quantities)
    
    # 주문 데이터 구조화
    order_info = {
        'order_id': order_id,
        'items': {name: qty for name, qty in zip(request.item_names, request.item_quantities)}
    }
    order_queue.put((priority, order_info))

    # 5. 응답 생성
    response.assigned_order_id = order_id
    response.message = "주문이 정상적으로 접수되었습니다."
    response.estimated_time = sum(request.item_quantities) * 10 # 개당 10초 잡음
    response.all_item_ids = list(inventory.keys())
    response.all_item_counts = list(inventory.values())

    node_.get_logger().info(f"✅ 주문 승인: {order_id} (예상대기: {response.estimated_time}초)")
    return response

# 2-2. 비전 토픽 콜백
def vision_callback(msg):
    global current_vision_data
    # 가장 최근 인식된 물체 정보 업데이트
    # node_.get_logger().info(f"[Vision] 감지됨: {msg.object_id}")
    current_vision_data = msg


def ros_spin_loop():
    global node_
    node_.get_logger().info("[ROS Thread] 통신 스레드 시작")
    
    # Service Server 생성
    srv = node_.create_service(OrderService, 'order_service', handle_order_request)
    
    # Vision Subscriber 생성
    sub = node_.create_subscription(DetectedObject, 'detected_object', vision_callback, 10)
    
    # 재고 현황 Publisher (옵션: 주기적으로 방송하고 싶을 때)
    # pub = node_.create_publisher(String, 'inventory_status', 10)

    rclpy.spin(node_)


# ==============================================================================
# [3] 메인 실행부
# ==============================================================================
def main(args=None):
    global node_
    
    rclpy.init(args=args)
    
    # 두산 로봇 네임스페이스에 맞춰 노드 생성
    node_ = rclpy.create_node("robot_controller_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = node_

    try:
        # 스레드 생성 (로봇 제어 / ROS 통신)
        t_robot = threading.Thread(target=perform_robot_task)
        t_ros = threading.Thread(target=ros_spin_loop)
        
        # 데몬 스레드로 설정 (메인 종료 시 같이 종료)
        t_robot.daemon = True
        t_ros.daemon = True
        
        t_robot.start()
        t_ros.start()
        
        # 메인 스레드는 유지
        t_robot.join()
        t_ros.join()

    except KeyboardInterrupt:
        node_.get_logger().info("종료 요청 받음.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()