#!/usr/bin/env python3
# dummy_vision_node.py

import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point, Quaternion
from scipy.spatial.transform import Rotation as R
import random

class DummyVisionNode(Node):
    def __init__(self):
        super().__init__('dummy_vision_node')
        # 로봇 컨트롤러가 찾는 네임스페이스와 동일하게 설정
        self.srv = self.create_service(DetectObject, '/dsr01/detect_object', self.handle_detect_object)
        self.get_logger().info("👀 가상 비전 센서 준비 완료 (Real Motion Ready)")

    def handle_detect_object(self, request, response):
        target_id = request.target_object_id
        self.get_logger().info(f"요청 수신: '{target_id}'")

        # [중요] 물체를 찾았다고 가정 (확률 100%로 설정하여 테스트 용이하게)
        response.found = True
        
        # 1. 위치 (Position)
        # 로봇 베이스 기준 좌표 (단위: mm)
        # x: 로봇 앞쪽 300~500mm
        # y: 좌우 -200~200mm
        # z: 0.0 (바닥 혹은 테이블 높이) -> 로봇이 집으러 내려갈 높이
        response.position = Point(
            x=random.uniform(250.0, 350.0), 
            y=random.uniform(250.0, 350.0), 
            z=250.0 
        )
        
        # 2. 자세 (Orientation)
        # 로봇이 그리퍼로 물건을 집으려면 '아래'를 봐야 합니다.
        # 두산 로봇 기준 (Euler XYZ): rx=0, ry=180, rz=0 이 아래를 보는 자세입니다.
        # 이를 쿼터니언으로 변환해서 보냅니다.
        rot = R.from_euler('xyz', [0, 180, 0], degrees=True)
        quat = rot.as_quat() # [x, y, z, w]
        
        response.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )
        
        response.confidence = 0.99
        response.message = f"Found {target_id} at {response.position.x:.1f}, {response.position.y:.1f}"
            
        return response

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