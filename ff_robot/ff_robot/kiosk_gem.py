#!/usr/bin/env python3
# kiosk_gem.py
#
# 주문 키오스크 시뮬레이터
# - 사용자로부터 메뉴 입력 받음 (CLI)
# - 메인 컨트롤러에게 주문 요청 (Service Client)
# - 재고 부족 시 예외 처리 메시지 출력

import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import OrderService
import sys

class KioskNode(Node):
    def __init__(self):
        super().__init__('kiosk_node')
        
        # 주문 서비스 클라이언트 설정
        self.cli = self.create_client(OrderService, '/dsr01/order_service')
        
        # 서비스 준비 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('메인 로봇 컨트롤러(Service) 연결 대기 중...')
            
        self.get_logger().info('✅ 키오스크 시스템 준비 완료. 주문 가능.')

    def send_order(self, item_names, item_quantities):
        """주문 요청 전송 함수"""
        req = OrderService.Request()
        req.item_names = item_names
        req.item_quantities = item_quantities
        
        self.get_logger().info(f"주문 전송 중... {item_names}")
        
        # 비동기 호출
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        return future.result()

def get_user_input():
    """사용자 CLI 입력 처리"""
    print("\n" + "="*40)
    print("      🍔 HAMBURGER KIOSK 🍟      ")
    print("="*40)
    # [수정] YOLO 학습 라벨과 일치하는 메뉴판으로 변경
    print("메뉴: burger1, burger2, burger3, fries, nugget, coke, cider")
    print("입력 형식: 메뉴1 수량1, 메뉴2 수량2 (종료: q)")
    print("예시: burger1 2, coke 1") # [수정] 예시도 변경
    
    user_in = input("\n주문 입력 > ").strip()
    if user_in.lower() == 'q':
        return None, None
        
    try:
        # 파싱 로직
        items = []
        quantities = []
        
        parts = user_in.split(',')
        for part in parts:
            menu, qty = part.strip().split()
            items.append(menu)
            quantities.append(int(qty))
            
        return items, quantities
    except Exception:
        print("⚠️ 입력 형식이 잘못되었습니다. 다시 시도해주세요.")
        return [], []

def main(args=None):
    rclpy.init(args=args)
    kiosk = KioskNode()

    try:
        while rclpy.ok():
            names, qtys = get_user_input()
            
            # 종료 조건
            if names is None: 
                break
                
            # 잘못된 입력 재시도
            if not names: 
                continue
                
            # 서비스 요청
            response = kiosk.send_order(names, qtys)
            
            # 결과 처리
            if response.assigned_order_id:
                print(f"\n✅ [주문 성공]")
                print(f"   - 주문번호: {response.assigned_order_id}")
                print(f"   - 안내메시지: {response.message}")
            else:
                print(f"\n🚫 [주문 실패]")
                print(f"   - 사유: {response.message}")
            
            # 재고 현황 표시
            print("\n   [현재 재고 현황]")
            for i, item_id in enumerate(response.all_item_ids):
                print(f"   - {item_id}: {response.all_item_counts[i]}")

    except KeyboardInterrupt:
        pass
    finally:
        kiosk.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()