#!/usr/bin/env python3
"""
로봇 상태 실시간 모니터링
조인트 상태, 위치, 속도 등을 출력합니다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class RobotStateMonitor(Node):
    def __init__(self):
        super().__init__('robot_state_monitor')
        
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.latest_state = None
        
        # 주기적으로 상태 출력
        self.timer = self.create_timer(2.0, self.print_status)
        
        self.get_logger().info("📊 로봇 상태 모니터링 시작!")
        
    def joint_state_callback(self, msg):
        """조인트 상태 업데이트"""
        self.latest_state = msg
        
    def print_status(self):
        """상태 출력"""
        if self.latest_state is None:
            self.get_logger().warn("⚠️ 조인트 상태를 아직 받지 못했습니다...")
            return
            
        print("\n" + "="*70)
        print("🤖 로봇 현재 상태")
        print("="*70)
        
        for i, name in enumerate(self.latest_state.name):
            pos = self.latest_state.position[i]
            vel = self.latest_state.velocity[i] if self.latest_state.velocity else 0.0
            
            # Radian to Degree 변환
            pos_deg = math.degrees(pos)
            vel_deg = math.degrees(vel)
            
            print(f"{name:20s} | Pos: {pos:7.3f} rad ({pos_deg:7.2f}°) | Vel: {vel:7.3f} rad/s")
        
        print("="*70 + "\n")

def main():
    rclpy.init()
    monitor = RobotStateMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info("\n👋 모니터링 종료")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
