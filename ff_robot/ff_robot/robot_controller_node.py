#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_controller_node.py

햄버거집 세팅 로봇의 메인 컨트롤러 노드
- 두산로봇 DR_init 패턴을 따르는 스크립트 형식
- 키오스크로부터 주문을 받아 처리
- 비전 노드와 통신하여 물체 위치 파악
- 로봇을 제어하여 Pick & Place 수행

Architecture:
- Service Server: /burger_robot/order_service (키오스크 주문 접수)
- Service Client: /burger_robot/get_target_pose (비전 노드에 물체 위치 요청)
- Service Client: /burger_robot/set_camera_mode (카메라 모드 전환)
- Topic Publisher: /burger_robot/inventory_status (재고 현황)
- Topic Publisher: /burger_robot/order_status (주문 상태)
"""

import rclpy
from rclpy.node import Node
import DR_init
import threading
import time
import random
from datetime import datetime
from collections import deque
from rclpy.executors import MultiThreadedExecutor

# 서비스 및 메시지 타입 (burger_interfaces 패키지)
# 빌드 전 테스트를 위해 try-except 처리
try:
    from burger_interfaces.srv import OrderService, GetTargetPose, SetCameraMode
    from burger_interfaces.msg import InventoryStatus, OrderStatus
    HAS_BURGER_INTERFACES = True
except ImportError:
    HAS_BURGER_INTERFACES = False
    print("[WARN] burger_interfaces 패키지를 찾을 수 없습니다. 테스트 모드로 실행합니다.")

# 표준 서비스/메시지
from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Point, Quaternion

# --- 로봇 설정 상수 (두산로봇) ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TCP = "Tool Weight"
ROBOT_TOOL = "GripperDA_v1"
VELOCITY = 60
ACC = 60

# --- DR_init 설정 ---
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# --- 전역 변수 ---
node_ = None
task_running = False
task_queue = deque()  # 작업 큐
task_event = threading.Event()

# 재고 관리
inventory = {
    "bulgogi_burger": {"count": 5, "name_kr": "불고기버거", "prep_time": 15},
    "cheese_burger": {"count": 5, "name_kr": "치즈버거", "prep_time": 12},
    "french_fries": {"count": 10, "name_kr": "감자튀김", "prep_time": 5},
    "coke": {"count": 8, "name_kr": "콜라", "prep_time": 3},
    "sprite": {"count": 8, "name_kr": "사이다", "prep_time": 3},
    "coffee": {"count": 6, "name_kr": "커피", "prep_time": 5}
}

# 주문 관리
order_counter = 0
active_orders = {}  # order_id -> order_info
order_queue = deque()  # 대기 중인 주문 ID

# 트레이 위치 (사전 정의된 배치 좌표)
TRAY_POSITIONS = {
    "slot_1": [300.0, 100.0, 200.0, 0.0, 180.0, 0.0],  # 버거용
    "slot_2": [300.0, 50.0, 200.0, 0.0, 180.0, 0.0],   # 사이드용
    "slot_3": [300.0, 0.0, 200.0, 0.0, 180.0, 0.0],    # 음료용
    "slot_4": [300.0, -50.0, 200.0, 0.0, 180.0, 0.0],  # 추가용
}

# 물체 유형별 트레이 슬롯 매핑
ITEM_SLOT_MAP = {
    "bulgogi_burger": "slot_1",
    "cheese_burger": "slot_1",
    "french_fries": "slot_2",
    "coke": "slot_3",
    "sprite": "slot_3",
    "coffee": "slot_3",
}

# 서비스 클라이언트 (비전 노드용)
cli_get_target_pose = None
cli_set_camera_mode = None

# 퍼블리셔
pub_inventory_status = None
pub_order_status = None

# --- 로봇 초기화 함수 ---
def initialize_robot():
    """두산로봇 초기화 (Tool, TCP 설정)"""
    global node_
    logger = node_.get_logger()
    
    try:
        from DSR_ROBOT2 import set_tool, set_tcp
    except ImportError as e:
        logger.warn(f"[WARN] DSR_ROBOT2 임포트 실패: {e}")
        logger.warn("[WARN] 시뮬레이션 모드로 진행합니다.")
        return True  # 시뮬레이션 모드에서는 성공 처리
    
    logger.info("로봇 초기 설정(Tool, TCP)을 시작합니다...")
    try:
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)
        logger.info("로봇 초기화 완료.")
        return True
    except Exception as e:
        logger.error(f"set_tool/set_tcp 실행 실패: {e}")
        return False


# --- 재고 관리 함수 ---
def check_inventory(item_names, item_quantities):
    """
    재고 확인 및 차감 가능 여부 체크
    
    Returns:
        (success: bool, message: str, unavailable_items: list)
    """
    global inventory
    unavailable = []
    
    for item, qty in zip(item_names, item_quantities):
        if item not in inventory:
            unavailable.append(f"알 수 없는 메뉴: {item}")
        elif inventory[item]["count"] < qty:
            unavailable.append(
                f"{inventory[item]['name_kr']}: 재고 {inventory[item]['count']}개, 요청 {qty}개"
            )
    
    if unavailable:
        return False, "재고 부족: " + ", ".join(unavailable), unavailable
    
    return True, "재고 확인 완료", []


def deduct_inventory(item_names, item_quantities):
    """재고 차감"""
    global inventory
    for item, qty in zip(item_names, item_quantities):
        if item in inventory:
            inventory[item]["count"] -= qty


def restore_inventory(item_names, item_quantities):
    """주문 취소 시 재고 복구"""
    global inventory
    for item, qty in zip(item_names, item_quantities):
        if item in inventory:
            inventory[item]["count"] += qty


def get_inventory_data():
    """현재 재고 정보 반환"""
    global inventory
    item_ids = list(inventory.keys())
    item_counts = [inventory[k]["count"] for k in item_ids]
    item_names_kr = [inventory[k]["name_kr"] for k in item_ids]
    item_available = [inventory[k]["count"] > 0 for k in item_ids]
    return item_ids, item_counts, item_names_kr, item_available


def calculate_estimated_time(item_names, item_quantities):
    """예상 준비 시간 계산"""
    global inventory, order_queue
    
    # 현재 주문의 기본 준비 시간
    base_time = 0
    for item, qty in zip(item_names, item_quantities):
        if item in inventory:
            base_time += inventory[item]["prep_time"] * qty
    
    # 앞선 대기 주문 시간 추가
    queue_wait = len(order_queue) * 30  # 주문당 평균 30초
    
    return base_time + queue_wait


# --- 주문 관리 함수 ---
def generate_order_id():
    """주문 번호 생성"""
    global order_counter
    order_counter += 1
    return f"ORD-{order_counter:03d}"


def create_order(order_id, item_names, item_quantities):
    """주문 생성 및 큐에 추가"""
    global active_orders, order_queue
    
    order_info = {
        "order_id": order_id,
        "items": list(zip(item_names, item_quantities)),
        "status": "QUEUED",
        "created_at": time.time(),
        "estimated_time": calculate_estimated_time(item_names, item_quantities),
    }
    
    active_orders[order_id] = order_info
    order_queue.append(order_id)
    
    return order_info


def get_queue_position(order_id):
    """대기열에서의 위치 반환 (1-based)"""
    global order_queue
    try:
        return list(order_queue).index(order_id) + 1
    except ValueError:
        return 0


# --- 상태 발행 함수 ---
def publish_inventory_status():
    """재고 현황 토픽 발행"""
    global node_, pub_inventory_status
    
    if not HAS_BURGER_INTERFACES or pub_inventory_status is None:
        return
    
    item_ids, item_counts, item_names_kr, item_available = get_inventory_data()
    
    msg = InventoryStatus()
    msg.header.stamp = node_.get_clock().now().to_msg()
    msg.item_ids = item_ids
    msg.item_counts = item_counts
    msg.item_names_kr = item_names_kr
    msg.item_available = item_available
    
    pub_inventory_status.publish(msg)


def publish_order_status(order_id, status, message=""):
    """주문 상태 토픽 발행"""
    global node_, pub_order_status, active_orders
    
    if not HAS_BURGER_INTERFACES or pub_order_status is None:
        return
    
    if order_id not in active_orders:
        return
    
    order_info = active_orders[order_id]
    
    msg = OrderStatus()
    msg.header.stamp = node_.get_clock().now().to_msg()
    msg.order_id = order_id
    msg.status = status
    msg.items = [f"{item}x{qty}" for item, qty in order_info["items"]]
    msg.queue_position = get_queue_position(order_id)
    msg.estimated_wait_sec = order_info["estimated_time"]
    msg.message = message
    
    pub_order_status.publish(msg)


# --- 서비스 콜백 ---
def handle_order_service(request, response):
    """
    주문 접수 서비스 콜백
    키오스크로부터 주문을 받아 처리
    """
    global node_, inventory, task_event
    logger = node_.get_logger()
    
    logger.info(f"[Order] 주문 요청 수신: {list(zip(request.item_names, request.item_quantities))}")
    
    # 1. 입력 유효성 검사
    if len(request.item_names) != len(request.item_quantities):
        response.success = False
        response.assigned_order_id = ""
        response.message = "오류: 메뉴와 수량 리스트 길이가 일치하지 않습니다."
        response.estimated_time_sec = 0
        response.queue_position = 0
        item_ids, item_counts, _, _ = get_inventory_data()
        response.all_item_ids = item_ids
        response.all_item_counts = item_counts
        logger.error(response.message)
        return response
    
    if len(request.item_names) == 0:
        response.success = False
        response.assigned_order_id = ""
        response.message = "오류: 주문 항목이 없습니다."
        response.estimated_time_sec = 0
        response.queue_position = 0
        item_ids, item_counts, _, _ = get_inventory_data()
        response.all_item_ids = item_ids
        response.all_item_counts = item_counts
        logger.warn(response.message)
        return response
    
    # 수량 유효성 검사
    for qty in request.item_quantities:
        if qty <= 0:
            response.success = False
            response.assigned_order_id = ""
            response.message = "오류: 수량은 1 이상이어야 합니다."
            response.estimated_time_sec = 0
            response.queue_position = 0
            item_ids, item_counts, _, _ = get_inventory_data()
            response.all_item_ids = item_ids
            response.all_item_counts = item_counts
            logger.warn(response.message)
            return response
    
    # 2. 재고 확인
    can_fulfill, message, unavailable = check_inventory(
        request.item_names, request.item_quantities
    )
    
    if not can_fulfill:
        response.success = False
        response.assigned_order_id = ""
        response.message = message
        response.estimated_time_sec = 0
        response.queue_position = 0
        item_ids, item_counts, _, _ = get_inventory_data()
        response.all_item_ids = item_ids
        response.all_item_counts = item_counts
        logger.warn(f"[Order] 재고 부족으로 주문 거절: {unavailable}")
        return response
    
    # 3. 재고 차감
    deduct_inventory(request.item_names, request.item_quantities)
    
    # 4. 주문 생성
    order_id = generate_order_id()
    order_info = create_order(order_id, request.item_names, request.item_quantities)
    
    # 5. 작업 큐에 추가
    task_queue.append({
        "type": "PROCESS_ORDER",
        "order_id": order_id,
        "items": list(zip(request.item_names, request.item_quantities))
    })
    task_event.set()  # 작업 스레드 깨우기
    
    # 6. 응답 구성
    response.success = True
    response.assigned_order_id = order_id
    response.message = f"주문이 접수되었습니다. 주문번호: {order_id}"
    response.estimated_time_sec = order_info["estimated_time"]
    response.queue_position = get_queue_position(order_id)
    
    item_ids, item_counts, _, _ = get_inventory_data()
    response.all_item_ids = item_ids
    response.all_item_counts = item_counts
    
    logger.info(f"[Order] 주문 접수 완료: {order_id}, 대기순서: {response.queue_position}")
    
    # 재고 현황 발행
    publish_inventory_status()
    publish_order_status(order_id, "QUEUED", "주문이 접수되었습니다.")
    
    return response


# --- 비전 노드 통신 함수 ---
def request_target_pose(object_id):
    """
    비전 노드에 물체 위치 요청
    
    Returns:
        (success, position, orientation, confidence, message)
    """
    global node_, cli_get_target_pose
    logger = node_.get_logger()
    
    if cli_get_target_pose is None:
        logger.warn("[Vision] 비전 서비스 클라이언트가 없습니다. 시뮬레이션 좌표 반환.")
        # 시뮬레이션 모드: 가상 좌표 반환
        pos = Point(x=400.0 + random.uniform(-50, 50), 
                    y=100.0 + random.uniform(-50, 50), 
                    z=100.0)
        orient = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return True, pos, orient, 0.95, "시뮬레이션 좌표"
    
    if not cli_get_target_pose.service_is_ready():
        logger.error("[Vision] get_target_pose 서비스가 준비되지 않았습니다.")
        return False, None, None, 0.0, "비전 서비스 연결 실패"
    
    request = GetTargetPose.Request()
    request.target_object_id = object_id
    
    logger.info(f"[Vision] '{object_id}' 위치 요청 중...")
    
    future = cli_get_target_pose.call_async(request)
    
    # 타임아웃 설정 (5초)
    timeout_sec = 5.0
    start_time = time.time()
    
    while not future.done():
        if time.time() - start_time > timeout_sec:
            logger.error("[Vision] 비전 서비스 응답 타임아웃")
            return False, None, None, 0.0, "비전 서비스 타임아웃"
        time.sleep(0.1)
    
    try:
        result = future.result()
        if result.found:
            logger.info(f"[Vision] 물체 발견: pos=({result.position.x:.1f}, {result.position.y:.1f}, {result.position.z:.1f})")
            return True, result.position, result.orientation, result.confidence, result.message
        else:
            logger.warn(f"[Vision] 물체를 찾을 수 없음: {object_id}")
            return False, None, None, 0.0, result.message
    except Exception as e:
        logger.error(f"[Vision] 서비스 호출 실패: {e}")
        return False, None, None, 0.0, str(e)


def set_camera_mode(mode):
    """카메라 모드 전환 요청"""
    global node_, cli_set_camera_mode
    logger = node_.get_logger()
    
    if cli_set_camera_mode is None:
        logger.warn(f"[Vision] 카메라 모드 서비스 없음. 모드 '{mode}' 설정 생략.")
        return True
    
    if not cli_set_camera_mode.service_is_ready():
        logger.error("[Vision] set_camera_mode 서비스가 준비되지 않았습니다.")
        return False
    
    request = SetCameraMode.Request()
    request.mode = mode
    
    logger.info(f"[Vision] 카메라 모드 전환 요청: {mode}")
    
    future = cli_set_camera_mode.call_async(request)
    
    timeout_sec = 3.0
    start_time = time.time()
    
    while not future.done():
        if time.time() - start_time > timeout_sec:
            logger.error("[Vision] 카메라 모드 전환 타임아웃")
            return False
        time.sleep(0.1)
    
    try:
        result = future.result()
        if result.success:
            logger.info(f"[Vision] 카메라 모드 전환 완료: {result.current_mode}")
            return True
        else:
            logger.error(f"[Vision] 카메라 모드 전환 실패: {result.message}")
            return False
    except Exception as e:
        logger.error(f"[Vision] 서비스 호출 실패: {e}")
        return False


# --- 로봇 동작 함수 (두산로봇용 - 실제 이동은 주석 처리) ---
def pick_object(object_id, position, orientation):
    """
    물체 파지 동작
    실제 두산로봇 명령어가 들어갈 위치
    """
    global node_
    logger = node_.get_logger()
    
    logger.info(f"[Robot] '{object_id}' 파지 시작")
    logger.info(f"[Robot] 목표 위치: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
    
    # ============================================================
    # [TODO] 여기에 실제 두산로봇 움직임 코드 삽입
    # 예시:
    # from DSR_ROBOT2 import (
    #     amovel, check_motion, posx, set_digital_output,
    #     DR_BASE, DR_MV_MOD_ABS, ON, OFF
    # )
    #
    # # 1. 물체 위로 이동
    # approach_pose = posx(position.x, position.y, position.z + 100, 0, 180, 0)
    # amovel(approach_pose, vel=VELOCITY, acc=ACC, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # wait_for_motion()
    #
    # # 2. 물체 위치로 하강
    # pick_pose = posx(position.x, position.y, position.z, 0, 180, 0)
    # amovel(pick_pose, vel=VELOCITY/2, acc=ACC/2, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # wait_for_motion()
    #
    # # 3. 그리퍼 닫기
    # set_digital_output(1, ON)
    # set_digital_output(2, OFF)
    # time.sleep(0.5)
    #
    # # 4. 상승
    # amovel(approach_pose, vel=VELOCITY, acc=ACC, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # wait_for_motion()
    # ============================================================
    
    # 시뮬레이션: 동작 시간 대기
    time.sleep(2.0)
    
    logger.info(f"[Robot] '{object_id}' 파지 완료")
    return True


def place_object(object_id, tray_slot):
    """
    물체 배치 동작
    실제 두산로봇 명령어가 들어갈 위치
    """
    global node_
    logger = node_.get_logger()
    
    if tray_slot not in TRAY_POSITIONS:
        logger.error(f"[Robot] 알 수 없는 트레이 슬롯: {tray_slot}")
        return False
    
    target_pose = TRAY_POSITIONS[tray_slot]
    logger.info(f"[Robot] '{object_id}' 배치 시작 -> {tray_slot}")
    logger.info(f"[Robot] 목표 위치: {target_pose}")
    
    # ============================================================
    # [TODO] 여기에 실제 두산로봇 움직임 코드 삽입
    # 예시:
    # from DSR_ROBOT2 import (
    #     amovel, check_motion, posx, set_digital_output,
    #     DR_BASE, DR_MV_MOD_ABS, ON, OFF
    # )
    #
    # # 1. 배치 위치 위로 이동
    # approach_pose = posx(target_pose[0], target_pose[1], target_pose[2] + 100,
    #                      target_pose[3], target_pose[4], target_pose[5])
    # amovel(approach_pose, vel=VELOCITY, acc=ACC, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # wait_for_motion()
    #
    # # 2. 배치 위치로 하강
    # place_pose = posx(*target_pose)
    # amovel(place_pose, vel=VELOCITY/2, acc=ACC/2, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # wait_for_motion()
    #
    # # 3. 그리퍼 열기
    # set_digital_output(1, OFF)
    # set_digital_output(2, ON)
    # time.sleep(0.5)
    #
    # # 4. 상승
    # amovel(approach_pose, vel=VELOCITY, acc=ACC, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # wait_for_motion()
    # ============================================================
    
    # 시뮬레이션: 동작 시간 대기
    time.sleep(2.0)
    
    logger.info(f"[Robot] '{object_id}' 배치 완료")
    return True


def move_to_home():
    """홈 포지션으로 이동"""
    global node_
    logger = node_.get_logger()
    
    logger.info("[Robot] 홈 포지션으로 이동")
    
    # ============================================================
    # [TODO] 여기에 실제 두산로봇 홈 이동 코드 삽입
    # 예시:
    # from DSR_ROBOT2 import movej, check_motion
    # home_joint = [0, 0, 90, 0, 90, 0]
    # movej(home_joint, vel=30, acc=30)
    # wait_for_motion()
    # ============================================================
    
    time.sleep(1.0)
    
    logger.info("[Robot] 홈 포지션 도착")
    return True


# --- 주문 처리 함수 ---
def process_order(order_id, items):
    """
    주문 처리 메인 로직
    items: [(item_name, quantity), ...]
    """
    global node_, active_orders, order_queue
    logger = node_.get_logger()
    
    logger.info(f"[Process] 주문 처리 시작: {order_id}")
    logger.info(f"[Process] 주문 항목: {items}")
    
    # 상태 업데이트: 처리 중
    if order_id in active_orders:
        active_orders[order_id]["status"] = "PROCESSING"
    publish_order_status(order_id, "PROCESSING", "주문을 준비 중입니다...")
    
    # 1. 카메라 모드 전환: 픽업 영역
    set_camera_mode("PICKUP_ZONE")
    time.sleep(0.5)
    
    # 2. 각 항목 처리
    all_success = True
    for item_name, quantity in items:
        for i in range(quantity):
            logger.info(f"[Process] {item_name} ({i+1}/{quantity}) 처리 중...")
            
            # 2.1. 비전으로 물체 위치 파악
            success, position, orientation, confidence, message = request_target_pose(item_name)
            
            if not success:
                logger.error(f"[Process] {item_name} 위치 파악 실패: {message}")
                # 재시도 로직
                retry_count = 0
                max_retries = 3
                while not success and retry_count < max_retries:
                    retry_count += 1
                    logger.warn(f"[Process] {item_name} 재시도 {retry_count}/{max_retries}")
                    time.sleep(1.0)
                    success, position, orientation, confidence, message = request_target_pose(item_name)
                
                if not success:
                    logger.error(f"[Process] {item_name} 최종 실패. 건너뜁니다.")
                    all_success = False
                    continue
            
            # 2.2. 신뢰도 체크
            if confidence < 0.7:
                logger.warn(f"[Process] {item_name} 감지 신뢰도 낮음: {confidence:.2f}")
            
            # 2.3. 물체 파지
            if not pick_object(item_name, position, orientation):
                logger.error(f"[Process] {item_name} 파지 실패")
                all_success = False
                continue
            
            # 2.4. 트레이에 배치
            tray_slot = ITEM_SLOT_MAP.get(item_name, "slot_4")
            if not place_object(item_name, tray_slot):
                logger.error(f"[Process] {item_name} 배치 실패")
                all_success = False
                continue
            
            logger.info(f"[Process] {item_name} ({i+1}/{quantity}) 완료")
    
    # 3. 홈 포지션으로 복귀
    move_to_home()
    
    # 4. 대기열에서 제거
    if order_id in order_queue:
        order_queue.remove(order_id)
    
    # 5. 상태 업데이트
    if all_success:
        if order_id in active_orders:
            active_orders[order_id]["status"] = "READY"
        publish_order_status(order_id, "READY", f"주문번호 {order_id} 준비 완료! 픽업해주세요.")
        logger.info(f"[Process] 주문 완료: {order_id}")
    else:
        if order_id in active_orders:
            active_orders[order_id]["status"] = "READY"  # 부분 완료도 READY 처리
        publish_order_status(order_id, "READY", f"주문번호 {order_id} 준비 완료 (일부 항목 누락)")
        logger.warn(f"[Process] 주문 부분 완료: {order_id}")
    
    return all_success


# --- 작업 스레드 ---
def task_worker_thread():
    """
    작업 처리 스레드
    큐에서 작업을 꺼내 순차적으로 처리
    """
    global task_running, task_queue, task_event, node_
    
    print("[Task Thread] 작업 스레드 시작. 대기 중...")
    
    while rclpy.ok():
        # 이벤트 대기 (타임아웃 1초)
        triggered = task_event.wait(timeout=1.0)
        
        if not rclpy.ok():
            break
        
        if triggered and len(task_queue) > 0:
            task_event.clear()
            
            while len(task_queue) > 0 and rclpy.ok():
                task = task_queue.popleft()
                task_running = True
                
                try:
                    if task["type"] == "PROCESS_ORDER":
                        process_order(task["order_id"], task["items"])
                except Exception as e:
                    if node_:
                        node_.get_logger().error(f"[Task Thread] 작업 처리 중 오류: {e}")
                finally:
                    task_running = False
    
    print("[Task Thread] 작업 스레드 종료.")


# --- ROS 스핀 스레드 ---
def ros_spin_thread():
    """ROS2 이벤트 처리 스레드"""
    global node_
    
    print("[ROS Thread] ROS Spin 스레드 시작...")
    
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node_)
        executor.spin()
    except Exception as e:
        if rclpy.ok():
            node_.get_logger().error(f"[ROS Thread] Spin 중 예외 발생: {e}")
    finally:
        print("[ROS Thread] ROS Spin 스레드 종료.")


# --- 재고 발행 타이머 ---
def inventory_timer_callback():
    """주기적으로 재고 현황 발행"""
    publish_inventory_status()


# --- 메인 함수 ---
def main(args=None):
    global node_, cli_get_target_pose, cli_set_camera_mode
    global pub_inventory_status, pub_order_status
    
    rclpy.init(args=args)
    
    # 드라이버 충돌 방지를 위한 랜덤 지연
    delay = random.uniform(0.1, 0.5)
    print(f"[Main] 드라이버 충돌 방지를 위해 {delay:.2f}초 대기...")
    time.sleep(delay)
    
    # 노드 생성
    node_ = rclpy.create_node("robot_controller_node", namespace="burger_robot")
    
    # [중요] DR_init에 노드 연결
    DR_init.__dsr__node = node_
    
    logger = node_.get_logger()
    logger.info("=" * 50)
    logger.info("  햄버거 로봇 컨트롤러 노드 시작")
    logger.info("=" * 50)
    
    # --- 퍼블리셔 생성 ---
    if HAS_BURGER_INTERFACES:
        pub_inventory_status = node_.create_publisher(
            InventoryStatus, 'inventory_status', 10
        )
        pub_order_status = node_.create_publisher(
            OrderStatus, 'order_status', 10
        )
        logger.info("[Init] 퍼블리셔 생성 완료")
    
    # --- 서비스 서버 생성 ---
    if HAS_BURGER_INTERFACES:
        order_service = node_.create_service(
            OrderService, 'order_service', handle_order_service
        )
        logger.info("[Init] 주문 서비스 서버 생성 완료: /burger_robot/order_service")
    
    # --- 서비스 클라이언트 생성 (비전 노드용) ---
    if HAS_BURGER_INTERFACES:
        cli_get_target_pose = node_.create_client(
            GetTargetPose, 'get_target_pose'
        )
        cli_set_camera_mode = node_.create_client(
            SetCameraMode, 'set_camera_mode'
        )
        logger.info("[Init] 비전 서비스 클라이언트 생성 완료")
    
    # --- 재고 발행 타이머 (5초 주기) ---
    inventory_timer = node_.create_timer(5.0, inventory_timer_callback)
    
    # --- 스레드 시작 ---
    spin_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    task_thread = threading.Thread(target=task_worker_thread, daemon=True)
    
    try:
        # 로봇 초기화
        if not initialize_robot():
            logger.warn("[Init] 로봇 초기화 실패. 시뮬레이션 모드로 진행합니다.")
        
        spin_thread.start()
        task_thread.start()
        
        logger.info("[Init] 모든 스레드 시작 완료")
        logger.info("[Init] 주문 대기 중...")
        
        # 초기 재고 발행
        publish_inventory_status()
        
        # 메인 스레드는 대기
        spin_thread.join()
        task_thread.join()
        
    except KeyboardInterrupt:
        logger.info("\nCtrl+C 감지. 노드를 종료합니다.")
    except Exception as e:
        logger.error(f"예기치 않은 오류 발생: {e}")
    finally:
        print("모든 스레드 종료 및 ROS2 종료 시작...")
        
        if rclpy.ok():
            rclpy.shutdown()
        
        print("ROS2가 종료되었습니다.")


if __name__ == "__main__":
    main()