#!/usr/bin/env python3
"""
MoveIt2 액션 클라이언트 테스트 (moveit_py 불필요)
순수 ROS2 액션만 사용합니다.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, 
    Constraints, 
    JointConstraint,
    PositionIKRequest,
    RobotState
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time

class MoveItActionTest(Node):
    def __init__(self):
        super().__init__('moveit_action_test')
        
        # MoveGroup 액션 클라이언트
        self.move_action_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )
        
        # 현재 조인트 상태 구독
        self.current_joint_state = None
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info("🤖 MoveIt 액션 클라이언트 초기화 중...")
        
        # 액션 서버 대기
        self.move_action_client.wait_for_server()
        self.get_logger().info("✅ MoveGroup 액션 서버 연결 완료!")
        
        # 조인트 상태 대기
        while self.current_joint_state is None:
            self.get_logger().info("⏳ 조인트 상태 대기 중...")
            rclpy.spin_once(self, timeout_sec=1.0)
        
        self.get_logger().info("✅ 초기화 완료!")
        
    def joint_state_callback(self, msg):
        """현재 조인트 상태 업데이트"""
        self.current_joint_state = msg
        
    def move_to_joint_values(self, joint_values, group_name='manipulator'):
        """
        특정 조인트 값으로 이동
        joint_values: [j1, j2, j3, j4, j5, j6]
        """
        self.get_logger().info(f"📍 목표 조인트 값: {joint_values}")
        
        # Goal 생성
        goal_msg = MoveGroup.Goal()
        
        # Request 설정
        goal_msg.request.group_name = group_name
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1
        
        # 시작 상태 (현재 상태)
        goal_msg.request.start_state.joint_state = self.current_joint_state
        goal_msg.request.start_state.is_diff = False
        
        # 목표 제약 조건 설정
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        goal_constraint = Constraints()
        for i, (name, value) in enumerate(zip(joint_names, joint_values)):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = value
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0
            goal_constraint.joint_constraints.append(joint_constraint)
        
        goal_msg.request.goal_constraints.append(goal_constraint)
        
        # Planning only (Execute는 별도)
        goal_msg.planning_options.plan_only = False  # Plan & Execute
        goal_msg.planning_options.planning_scene_diff.is_diff = True
        goal_msg.planning_options.planning_scene_diff.robot_state.is_diff = True
        
        # 액션 전송
        self.get_logger().info("🚀 Planning & Execute 시작...")
        future = self.move_action_client.send_goal_async(goal_msg)
        
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("❌ Goal이 거부되었습니다!")
            return False
        
        self.get_logger().info("✅ Goal 수락됨! 실행 대기 중...")
        
        # 결과 대기
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        
        if result.error_code.val == 1:  # SUCCESS
            self.get_logger().info("✅ 성공!")
            return True
        else:
            self.get_logger().error(f"❌ 실패! Error code: {result.error_code.val}")
            return False

def main():
    rclpy.init()
    node = MoveItActionTest()
    
    try:
        # 초기화 대기
        time.sleep(1.0)
        
        # 테스트 시퀀스
        test_poses = [
            {
                'name': 'Home',
                'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            {
                'name': 'Pose 1',
                'values': [0.5, -0.5, 0.5, 0.0, 0.5, 0.0]
            },
            {
                'name': 'Pose 2',
                'values': [-0.5, -0.5, 0.5, 0.0, 0.5, 0.0]
            },
            {
                'name': 'Home',
                'values': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
        ]
        
        for pose in test_poses:
            node.get_logger().info(f"\n{'='*50}")
            node.get_logger().info(f"🎯 목표: {pose['name']}")
            node.get_logger().info(f"{'='*50}")
            
            success = node.move_to_joint_values(pose['values'])
            
            if success:
                time.sleep(2.0)  # 다음 동작 전 대기
            else:
                node.get_logger().warn("⚠️ 실패했지만 계속 진행합니다...")
                time.sleep(1.0)
        
        node.get_logger().info("\n✅ 모든 테스트 완료!")
        
    except KeyboardInterrupt:
        node.get_logger().info("\n⚠️ 사용자가 중단했습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()