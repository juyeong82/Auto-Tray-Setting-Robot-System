#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kiosk_node.py

햄버거집 키오스크 인터페이스 노드
- 사용자 주문 UI 제공 (터미널 기반)
- 음성 인식 주문 지원 (선택적)
- 로봇 컨트롤러에 주문 전송
- 재고 현황 실시간 표시
- 주문 상태 모니터링

Architecture:
- Service Client: /burger_robot/order_service (주문 전송)
- Topic Subscriber: /burger_robot/inventory_status (재고 수신)
- Topic Subscriber: /burger_robot/order_status (주문 상태 수신)
"""

import rclpy
from rclpy.node import Node
import threading
import time
import sys
import os
from datetime import datetime

# 서비스 및 메시지 타입
try:
    from burger_interfaces.srv import OrderService
    from burger_interfaces.msg import InventoryStatus, OrderStatus
    HAS_BURGER_INTERFACES = True
except ImportError:
    HAS_BURGER_INTERFACES = False
    print("[WARN] burger_interfaces 패키지를 찾을 수 없습니다. 테스트 모드로 실행합니다.")

# 음성 인식 모듈 (선택적)
try:
    from voice_command_processor_ham import InputHandler, ExtractKeyword
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False
    print("[INFO] 음성 인식 모듈을 사용할 수 없습니다. 텍스트 모드만 지원합니다.")


# --- 전역 변수 ---
node_ = None
inventory_data = {}  # item_id -> {count, name_kr, available}
order_history = {}   # order_id -> {status, items, message}
current_order_id = None

# 서비스 클라이언트
cli_order_service = None

# 실행 플래그
running = True


# --- 화면 출력 함수 ---
def clear_screen():
    """화면 지우기"""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_header():
    """헤더 출력"""
    print("=" * 60)
    print("    🍔 스마트 버거 키오스크 v1.0 🍔")
    print("=" * 60)
    print(f"    현재 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)


def print_menu():
    """메뉴 및 재고 출력"""
    global inventory_data
    
    print("\n📋 [ 메뉴 ]")
    print("-" * 40)
    
    if not inventory_data:
        print("  (재고 정보 로딩 중...)")
    else:
        # 카테고리별 정렬
        burgers = []
        sides = []
        drinks = []
        
        for item_id, info in inventory_data.items():
            if "burger" in item_id:
                burgers.append((item_id, info))
            elif "fries" in item_id:
                sides.append((item_id, info))
            else:
                drinks.append((item_id, info))
        
        # 버거
        print("  🍔 버거")
        for item_id, info in burgers:
            status = f"재고: {info['count']}" if info['available'] else "품절"
            status_color = "" if info['available'] else "(❌ 품절)"
            print(f"    - {info['name_kr']:10s} [{item_id:15s}] {status:10s} {status_color}")
        
        # 사이드
        print("  🍟 사이드")
        for item_id, info in sides:
            status = f"재고: {info['count']}" if info['available'] else "품절"
            status_color = "" if info['available'] else "(❌ 품절)"
            print(f"    - {info['name_kr']:10s} [{item_id:15s}] {status:10s} {status_color}")
        
        # 음료
        print("  🥤 음료")
        for item_id, info in drinks:
            status = f"재고: {info['count']}" if info['available'] else "품절"
            status_color = "" if info['available'] else "(❌ 품절)"
            print(f"    - {info['name_kr']:10s} [{item_id:15s}] {status:10s} {status_color}")
    
    print("-" * 40)


def print_order_status():
    """현재 주문 상태 출력"""
    global order_history, current_order_id
    
    if not order_history:
        return
    
    print("\n📦 [ 내 주문 현황 ]")
    print("-" * 40)
    
    for order_id, info in order_history.items():
        status_emoji = {
            "QUEUED": "⏳",
            "PROCESSING": "🔄",
            "READY": "✅",
            "SERVED": "🍽️",
            "CANCELLED": "❌"
        }.get(info['status'], "❓")
        
        print(f"  {status_emoji} 주문번호: {order_id}")
        print(f"     상태: {info['status']}")
        print(f"     항목: {', '.join(info['items'])}")
        if info['message']:
            print(f"     메시지: {info['message']}")
        print()
    
    print("-" * 40)


def print_commands():
    """명령어 안내 출력"""
    print("\n💡 [ 명령어 ]")
    print("-" * 40)
    print("  1. 주문하기     - 메뉴를 선택하여 주문")
    if HAS_VOICE:
        print("  2. 음성 주문    - 음성으로 주문")
    print("  3. 주문 확인    - 내 주문 상태 확인")
    print("  4. 새로고침     - 화면 새로고침")
    print("  0. 종료         - 키오스크 종료")
    print("-" * 40)


# --- 재고 콜백 ---
def inventory_callback(msg):
    """재고 현황 수신 콜백"""
    global inventory_data
    
    inventory_data.clear()
    
    for i, item_id in enumerate(msg.item_ids):
        inventory_data[item_id] = {
            "count": msg.item_counts[i],
            "name_kr": msg.item_names_kr[i] if i < len(msg.item_names_kr) else item_id,
            "available": msg.item_available[i] if i < len(msg.item_available) else (msg.item_counts[i] > 0)
        }


# --- 주문 상태 콜백 ---
def order_status_callback(msg):
    """주문 상태 수신 콜백"""
    global order_history
    
    order_history[msg.order_id] = {
        "status": msg.status,
        "items": list(msg.items),
        "message": msg.message,
        "queue_position": msg.queue_position,
        "estimated_wait": msg.estimated_wait_sec
    }
    
    # 준비 완료 알림
    if msg.status == "READY":
        print(f"\n🔔 알림: 주문번호 {msg.order_id} 준비 완료!")
        print(f"   {msg.message}")
        print()


# --- 주문 처리 함수 ---
def send_order(item_names, item_quantities):
    """
    주문 전송
    
    Args:
        item_names: 메뉴 ID 리스트
        item_quantities: 수량 리스트
    
    Returns:
        (success, order_id, message)
    """
    global node_, cli_order_service, order_history
    
    if cli_order_service is None:
        print("[ERROR] 주문 서비스가 연결되지 않았습니다.")
        return False, "", "서비스 연결 실패"
    
    # 서비스 준비 확인
    if not cli_order_service.service_is_ready():
        print("[WARN] 주문 서비스 연결 대기 중...")
        if not cli_order_service.wait_for_service(timeout_sec=5.0):
            print("[ERROR] 주문 서비스에 연결할 수 없습니다.")
            return False, "", "서비스 연결 타임아웃"
    
    # 요청 생성
    request = OrderService.Request()
    request.item_names = item_names
    request.item_quantities = item_quantities
    
    print("\n📤 주문 전송 중...")
    
    # 서비스 호출
    future = cli_order_service.call_async(request)
    
    # 응답 대기 (타임아웃 10초)
    timeout_sec = 10.0
    start_time = time.time()
    
    while not future.done():
        if time.time() - start_time > timeout_sec:
            print("[ERROR] 주문 응답 타임아웃")
            return False, "", "응답 타임아웃"
        time.sleep(0.1)
    
    try:
        result = future.result()
        
        if result.success:
            # 재고 정보 업데이트
            for i, item_id in enumerate(result.all_item_ids):
                if item_id in inventory_data:
                    inventory_data[item_id]["count"] = result.all_item_counts[i]
                    inventory_data[item_id]["available"] = result.all_item_counts[i] > 0
            
            # 주문 기록 추가
            order_history[result.assigned_order_id] = {
                "status": "QUEUED",
                "items": [f"{name}x{qty}" for name, qty in zip(item_names, item_quantities)],
                "message": result.message,
                "queue_position": result.queue_position,
                "estimated_wait": result.estimated_time_sec
            }
            
            return True, result.assigned_order_id, result.message
        else:
            return False, "", result.message
            
    except Exception as e:
        print(f"[ERROR] 주문 처리 실패: {e}")
        return False, "", str(e)


def text_order_flow():
    """텍스트 기반 주문 플로우"""
    global inventory_data
    
    print("\n" + "=" * 40)
    print("  📝 주문하기")
    print("=" * 40)
    
    if not inventory_data:
        print("[WARN] 재고 정보가 없습니다. 잠시 후 다시 시도해주세요.")
        return
    
    # 사용 가능한 메뉴 목록 출력
    available_items = {k: v for k, v in inventory_data.items() if v['available']}
    
    if not available_items:
        print("[WARN] 현재 주문 가능한 메뉴가 없습니다.")
        return
    
    print("\n주문 가능한 메뉴:")
    for i, (item_id, info) in enumerate(available_items.items(), 1):
        print(f"  {i}. {info['name_kr']} ({item_id}) - 재고: {info['count']}")
    
    print("\n메뉴 ID와 수량을 입력하세요.")
    print("예: bulgogi_burger 2, french_fries 1")
    print("(취소하려면 'q' 입력)")
    
    user_input = input("\n주문 입력 > ").strip()
    
    if user_input.lower() == 'q':
        print("주문이 취소되었습니다.")
        return
    
    # 입력 파싱
    try:
        items = []
        quantities = []
        
        # 쉼표로 분리
        parts = [p.strip() for p in user_input.split(',')]
        
        for part in parts:
            tokens = part.split()
            if len(tokens) >= 2:
                item_name = tokens[0]
                quantity = int(tokens[1])
                items.append(item_name)
                quantities.append(quantity)
            elif len(tokens) == 1:
                items.append(tokens[0])
                quantities.append(1)
        
        if not items:
            print("[ERROR] 유효한 주문 항목이 없습니다.")
            return
        
        # 유효성 검사
        for item in items:
            if item not in inventory_data:
                print(f"[ERROR] 알 수 없는 메뉴: {item}")
                return
            if not inventory_data[item]['available']:
                print(f"[ERROR] 품절된 메뉴: {inventory_data[item]['name_kr']}")
                return
        
        # 주문 확인
        print("\n📋 주문 내역:")
        for item, qty in zip(items, quantities):
            name_kr = inventory_data[item]['name_kr']
            print(f"  - {name_kr} x {qty}")
        
        confirm = input("\n주문하시겠습니까? (y/n) > ").strip().lower()
        
        if confirm == 'y':
            success, order_id, message = send_order(items, quantities)
            
            if success:
                print("\n" + "=" * 40)
                print(f"  ✅ 주문 완료!")
                print(f"  📋 주문번호: {order_id}")
                print(f"  💬 {message}")
                print("=" * 40)
            else:
                print("\n" + "=" * 40)
                print(f"  ❌ 주문 실패")
                print(f"  💬 {message}")
                print("=" * 40)
        else:
            print("주문이 취소되었습니다.")
            
    except ValueError as e:
        print(f"[ERROR] 입력 형식이 올바르지 않습니다: {e}")
    except Exception as e:
        print(f"[ERROR] 주문 처리 중 오류: {e}")


def voice_order_flow():
    """음성 기반 주문 플로우"""
    global inventory_data
    
    if not HAS_VOICE:
        print("[ERROR] 음성 인식 모듈을 사용할 수 없습니다.")
        return
    
    print("\n" + "=" * 40)
    print("  🎤 음성 주문")
    print("=" * 40)
    
    try:
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            return
        
        input_handler = InputHandler(api_key)
        keyword_extractor = ExtractKeyword(api_key)
        
        print("\n음성 또는 텍스트로 주문해주세요.")
        print("예: '불고기버거 하나랑 콜라 두 개 주세요'")
        
        # 사용자 입력 받기
        user_command = input_handler.get_input()
        
        if not user_command:
            print("[ERROR] 입력을 인식할 수 없습니다.")
            return
        
        print(f"\n인식된 명령: {user_command}")
        
        # 키워드 추출
        objects, destinations = keyword_extractor.run(user_command)
        
        if not objects:
            print("[ERROR] 주문 항목을 인식할 수 없습니다.")
            return
        
        print(f"추출된 메뉴: {objects}")
        
        # 메뉴 ID 매핑 및 수량 집계
        item_counts = {}
        for obj in objects:
            if obj in inventory_data:
                item_counts[obj] = item_counts.get(obj, 0) + 1
            else:
                print(f"[WARN] 알 수 없는 메뉴 무시: {obj}")
        
        if not item_counts:
            print("[ERROR] 유효한 주문 항목이 없습니다.")
            return
        
        items = list(item_counts.keys())
        quantities = list(item_counts.values())
        
        # 주문 확인
        print("\n📋 주문 내역:")
        for item, qty in zip(items, quantities):
            name_kr = inventory_data.get(item, {}).get('name_kr', item)
            print(f"  - {name_kr} x {qty}")
        
        confirm = input("\n주문하시겠습니까? (y/n) > ").strip().lower()
        
        if confirm == 'y':
            success, order_id, message = send_order(items, quantities)
            
            if success:
                print("\n" + "=" * 40)
                print(f"  ✅ 주문 완료!")
                print(f"  📋 주문번호: {order_id}")
                print(f"  💬 {message}")
                print("=" * 40)
            else:
                print("\n" + "=" * 40)
                print(f"  ❌ 주문 실패")
                print(f"  💬 {message}")
                print("=" * 40)
        else:
            print("주문이 취소되었습니다.")
            
    except Exception as e:
        print(f"[ERROR] 음성 주문 처리 중 오류: {e}")


# --- 메인 UI 루프 ---
def ui_loop():
    """메인 UI 루프"""
    global running
    
    while running:
        try:
            # 화면 출력
            clear_screen()
            print_header()
            print_menu()
            print_order_status()
            print_commands()
            
            # 사용자 입력
            choice = input("\n선택 > ").strip()
            
            if choice == '1':
                text_order_flow()
                input("\n계속하려면 Enter를 누르세요...")
                
            elif choice == '2' and HAS_VOICE:
                voice_order_flow()
                input("\n계속하려면 Enter를 누르세요...")
                
            elif choice == '3':
                print_order_status()
                input("\n계속하려면 Enter를 누르세요...")
                
            elif choice == '4':
                print("새로고침 중...")
                time.sleep(0.5)
                
            elif choice == '0':
                print("\n키오스크를 종료합니다. 감사합니다! 🙏")
                running = False
                break
                
            else:
                print("[WARN] 올바른 명령어를 입력해주세요.")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n키오스크를 종료합니다.")
            running = False
            break
        except Exception as e:
            print(f"[ERROR] UI 오류: {e}")
            time.sleep(1)


# --- ROS 스핀 스레드 ---
def ros_spin_thread():
    """ROS2 이벤트 처리 스레드"""
    global node_, running
    
    print("[ROS Thread] ROS Spin 스레드 시작...")
    
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node_, timeout_sec=0.1)
    except Exception as e:
        print(f"[ROS Thread] 오류: {e}")
    finally:
        print("[ROS Thread] ROS Spin 스레드 종료.")


# --- 메인 함수 ---
def main(args=None):
    global node_, cli_order_service, running
    
    rclpy.init(args=args)
    
    # 노드 생성
    node_ = rclpy.create_node("kiosk_node", namespace="burger_robot")
    
    logger = node_.get_logger()
    logger.info("키오스크 노드 시작")
    
    # --- 서비스 클라이언트 생성 ---
    if HAS_BURGER_INTERFACES:
        cli_order_service = node_.create_client(
            OrderService, 'order_service'
        )
        logger.info("주문 서비스 클라이언트 생성 완료")
        
        # 서비스 연결 대기
        print("주문 서비스 연결 대기 중...")
        if cli_order_service.wait_for_service(timeout_sec=5.0):
            print("✅ 주문 서비스 연결 완료")
        else:
            print("⚠️ 주문 서비스 연결 실패 - 오프라인 모드로 실행")
    
    # --- 구독자 생성 ---
    if HAS_BURGER_INTERFACES:
        inventory_sub = node_.create_subscription(
            InventoryStatus,
            'inventory_status',
            inventory_callback,
            10
        )
        
        order_status_sub = node_.create_subscription(
            OrderStatus,
            'order_status',
            order_status_callback,
            10
        )
        logger.info("토픽 구독자 생성 완료")
    
    # --- 스레드 시작 ---
    spin_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    spin_thread.start()
    
    # 재고 정보 수신 대기
    print("재고 정보 수신 대기 중...")
    time.sleep(2.0)
    
    try:
        # UI 루프 시작
        ui_loop()
        
    except KeyboardInterrupt:
        print("\n\n키오스크를 종료합니다.")
    finally:
        running = False
        
        if rclpy.ok():
            node_.destroy_node()
            rclpy.shutdown()
        
        print("키오스크가 종료되었습니다.")


if __name__ == "__main__":
    main()