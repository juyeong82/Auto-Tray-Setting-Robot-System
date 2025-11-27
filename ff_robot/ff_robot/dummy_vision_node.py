#!/usr/bin/env python3
# dummy_vision_node.py
#
# 비전 시스템이 준비되기 전, 가짜 감지 데이터를 발행하는 테스트 노드
# - 주기적으로 햄버거, 콜라 등의 위치 정보를 메인 컨트롤러에 전송

import rclpy
from rclpy.node import Node
from ff_robot_interfaces.msg import DetectedObject
from geometry_msgs.msg import Point
import time
import random

class DummyVisionNode(Node):
    def __init__(self):
        super().__init__('dummy_vision_node')
        
        # 메인 컨트롤러가 구독 중인 토픽 이름과 맞춰야 함 (/dsr01/detected_object)
        self.pub = self.create_publisher(DetectedObject, '/dsr01/detected_object', 10)
        
        # 1초마다 데이터 발행
        self.timer = self.create_timer(1.0, self.publish_fake_data)
        
        # 테스트용 물체 리스트
        self.test_objects = ['burger_cheese', 'coke', 'fries']
        self.get_logger().info("👀 가상 비전 센서 가동 시작...")

    def publish_fake_data(self):
        msg = DetectedObject()
        
        # 랜덤하게 물체 하나 선택
        msg.object_id = random.choice(self.test_objects)
        
        # 가짜 좌표 생성 (로봇 베이스 기준)
        msg.position = Point(
            x=random.uniform(300.0, 500.0), 
            y=random.uniform(-200.0, 200.0), 
            z=0.0
        )
        msg.confidence = 0.95 # 신뢰도 95%
        
        self.pub.publish(msg)
        # self.get_logger().info(f"데이터 전송: {msg.object_id} at ({msg.position.x:.1f}, {msg.position.y:.1f})")

def main(args=None):
    rclpy.init(args=args)
    node = DummyVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()