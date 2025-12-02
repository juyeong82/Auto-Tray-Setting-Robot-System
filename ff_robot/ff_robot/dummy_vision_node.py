#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import DetectObject
from geometry_msgs.msg import Point

class FakeVisionNode(Node):
    def __init__(self):
        super().__init__('fake_vision_node')
        
        # 실제 로봇 컨트롤러가 요청하는 서비스 이름과 동일해야 함
        self.srv = self.create_service(DetectObject, '/dsr01/detect_object', self.detect_callback)
        
        self.get_logger().info("👀 [Fake Vision] 가짜 비전 노드 시작 (YOLO 모사 중)")
        self.get_logger().info("   - 요청이 오면 가제보 환경에 맞는 고정 좌표를 반환합니다.")

        # =========================================================
        # [📍 좌표 데이터베이스] 가제보 월드(hamburger_station.world)와 일치시킴
        # 단위: mm (로봇 베이스 기준 상대 좌표로 변환됨)
        # 로봇 베이스 높이가 0.4m이므로, 물체 높이 0.82m는 로봇 기준 Z=420mm 정도임
        # =========================================================
        self.known_locations = {
            "burger1": {"x": 400.0, "y": 500.0, "z": 420.0, "rz": 0.0},
            "fries":   {"x": 400.0, "y": 300.0, "z": 420.0, "rz": 0.0},
            "coke":    {"x": 400.0, "y": 150.0, "z": 420.0, "rz": 0.0},
            "tray":    {"x": 600.0, "y": 0.0,   "z": 390.0, "rz": 90.0} # 트레이 중심
        }

    def detect_callback(self, request, response):
        target_name = request.target_name.lower()
        self.get_logger().info(f"📨 요청 수신: '{target_name}' 찾는 중...")

        if target_name in self.known_locations:
            loc = self.known_locations[target_name]
            
            response.found = True
            response.position = Point(x=loc['x'], y=loc['y'], z=loc['z'])
            
            # 오일러 각도 (Roll, Pitch는 고정, Yaw만 변경)
            response.rx = 0.0
            response.ry = 180.0 # 그리퍼가 아래를 보도록
            response.rz = loc['rz']
            
            response.width = 80.0  # 가짜 너비 (mm)
            response.height = 80.0 # 가짜 높이 (mm)
            response.confidence = 0.99
            
            self.get_logger().info(f"   ✅ 발견! 좌표 전송: X={loc['x']}, Y={loc['y']}")
        else:
            response.found = False
            self.get_logger().warn(f"   ❌ '{target_name}' 위치 정보 없음")

        return response

def main(args=None):
    rclpy.init(args=args)
    node = FakeVisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()