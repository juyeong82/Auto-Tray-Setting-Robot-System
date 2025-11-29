#!/usr/bin/env python3
"""
간단한 조인트 제어 테스트
각 조인트를 순차적으로 움직여 봅니다.
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class SimpleJointTest(Node):
    def __init__(self):
        super().__init__('simple_joint_test')
        
        # Publisher 생성
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/manipulator_controller/joint_trajectory',
            10
        )
        
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            10
        )
        
        self.get_logger().info("🤖 간단한 조인트 테스트 시작!")
        
    def move_arm_to_position(self, positions, duration_sec=3.0):
        """
        로봇 팔을 특정 위치로 이동
        positions: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
        """
        msg = JointTrajectory()
        msg.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=0)
        
        msg.points.append(point)
        
        self.arm_pub.publish(msg)
        self.get_logger().info(f"✅ 팔 이동 명령 전송: {positions}")
        
    def move_gripper(self, position, duration_sec=2.0):
        """
        그리퍼 제어
        position: 0.0 (닫힘) ~ 0.055 (완전 열림)
        """
        msg = JointTrajectory()
        msg.joint_names = ['rg2_finger_joint1']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=0)
        
        msg.points.append(point)
        
        self.gripper_pub.publish(msg)
        status = "열림" if position > 0.02 else "닫힘"
        self.get_logger().info(f"✅ 그리퍼 {status}: {position:.3f}")

def main():
    rclpy.init()
    node = SimpleJointTest()
    
    try:
        # 초기화 대기
        time.sleep(1.0)
        
        # 테스트 시퀀스
        node.get_logger().info("\n=== 테스트 1: Home 자세 ===")
        node.move_arm_to_position([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=3.0)
        time.sleep(4.0)
        
        node.get_logger().info("\n=== 테스트 2: 각 조인트 개별 이동 ===")
        # Joint 1 이동
        node.move_arm_to_position([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=2.0)
        time.sleep(3.0)
        
        # Joint 2 이동
        node.move_arm_to_position([0.5, -0.5, 0.0, 0.0, 0.0, 0.0], duration_sec=2.0)
        time.sleep(3.0)
        
        # Joint 3 이동
        node.move_arm_to_position([0.5, -0.5, 0.5, 0.0, 0.0, 0.0], duration_sec=2.0)
        time.sleep(3.0)
        
        node.get_logger().info("\n=== 테스트 3: 그리퍼 제어 ===")
        # 그리퍼 열기
        node.move_gripper(0.055, duration_sec=2.0)
        time.sleep(3.0)
        
        # 그리퍼 닫기
        node.move_gripper(0.0, duration_sec=2.0)
        time.sleep(3.0)
        
        node.get_logger().info("\n=== 테스트 4: Home으로 복귀 ===")
        node.move_arm_to_position([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=3.0)
        time.sleep(4.0)
        
        node.get_logger().info("\n✅ 모든 테스트 완료!")
        
    except KeyboardInterrupt:
        node.get_logger().info("\n⚠️ 사용자가 중단했습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
