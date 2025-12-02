#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import OrderService2
import json

class OrderTestNode(Node):
    def __init__(self):
        super().__init__('order_test_node')
        self.cli = self.create_client(OrderService2, '/dsr01/order_service')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ 로봇 컨트롤러(OrderService) 대기 중...')
            
        self.send_order()

    def send_order(self):
        req = OrderService2.Request()
        
        # ==============================
        # [🍔 테스트 주문 내용]
        # ==============================
        order_dict = {
            "burger1": 1,
            "fries": 1
        }
        # ==============================
        
        req.order_data_json = json.dumps(order_dict)
        req.order_id = "TEST_ORDER_001"
        req.table_id = "Table_1"
        
        self.get_logger().info(f"📤 주문 전송: {order_dict}")
        
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        result = future.result()
        if result.success:
            self.get_logger().info(f"✅ 주문 접수 성공: {result.message}")
        else:
            self.get_logger().error(f"❌ 주문 접수 실패: {result.message}")

def main():
    rclpy.init()
    node = OrderTestNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()