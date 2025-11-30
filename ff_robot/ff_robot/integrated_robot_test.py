#!/usr/bin/env python3
"""
완전 통합 테스트 스크립트
- 조인트 직접 제어
- 그리퍼 제어
- 미리 정의된 시나리오
의존성: 순수 ROS2만 사용 (moveit_py 불필요)
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import time
import math

class IntegratedRobotTest(Node):
    def __init__(self):
        super().__init__('integrated_robot_test')
        
        # Publishers
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
        
        # 현재 상태 구독
        self.current_joint_state = None
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info("🤖 통합 로봇 테스트 초기화 중...")
        
        # 조인트 상태 대기
        timeout = 5.0
        start = time.time()
        while self.current_joint_state is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        if self.current_joint_state is None:
            self.get_logger().error("❌ 조인트 상태를 받지 못했습니다!")
        else:
            self.get_logger().info("✅ 초기화 완료!")
        
        # 미리 정의된 자세들
        self.poses = {
            'home': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'ready': [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],
            'left': [0.8, -0.5, 0.5, 0.0, 0.5, 0.0],
            'right': [-0.8, -0.5, 0.5, 0.0, 0.5, 0.0],
            'up': [0.0, -0.3, 0.3, 0.0, 0.8, 0.0],
            'pickup': [0.5, -0.7, 0.8, 0.0, 0.5, 0.0],
            'place': [-0.5, -0.6, 0.7, 0.0, 0.5, 0.0],
        }
        
    def joint_state_callback(self, msg):
        """조인트 상태 업데이트"""
        self.current_joint_state = msg
        
    def get_current_positions(self):
        """현재 조인트 위치 가져오기"""
        if self.current_joint_state is None:
            return None
            
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        positions = []
        
        for name in joint_names:
            try:
                idx = self.current_joint_state.name.index(name)
                positions.append(self.current_joint_state.position[idx])
            except ValueError:
                positions.append(0.0)
        
        return positions
        
    def move_arm(self, pose_name_or_values, duration_sec=3.0, wait=True):
        """
        팔 이동
        pose_name_or_values: 'home' 같은 이름 또는 [j1,j2,j3,j4,j5,j6] 리스트
        """
        if isinstance(pose_name_or_values, str):
            if pose_name_or_values not in self.poses:
                self.get_logger().error(f"❌ 알 수 없는 자세: {pose_name_or_values}")
                return False
            positions = self.poses[pose_name_or_values]
            self.get_logger().info(f"🤖 '{pose_name_or_values}' 자세로 이동 중...")
        else:
            positions = pose_name_or_values
            self.get_logger().info(f"🤖 목표 위치로 이동 중: {positions}")
            
        msg = JointTrajectory()
        msg.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        msg.points.append(point)
        
        self.arm_pub.publish(msg)
        
        if wait:
            time.sleep(duration_sec + 0.5)
            
        return True
        
    def move_gripper(self, position_or_command, duration_sec=1.5, wait=True):
        """
        그리퍼 제어
        position_or_command: 'open', 'close', 또는 0.0~0.055 값
        """
        if position_or_command == 'open':
            position = 0.155
            status = "열기"
        elif position_or_command == 'close':
            position = 0.0
            status = "닫기"
        else:
            position = float(position_or_command)
            status = f"{position:.3f}"
            
        msg = JointTrajectory()
        msg.joint_names = ['rg2_finger_joint1']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        msg.points.append(point)
        
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"🤏 그리퍼 {status}")
        
        if wait:
            time.sleep(duration_sec + 0.5)
            
        return True
        
    def print_current_state(self):
        """현재 상태 출력"""
        positions = self.get_current_positions()
        if positions is None:
            self.get_logger().warn("⚠️ 현재 상태를 가져올 수 없습니다.")
            return
            
        print("\n" + "="*60)
        print("📊 현재 조인트 상태")
        print("="*60)
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        for name, pos in zip(joint_names, positions):
            deg = math.degrees(pos)
            print(f"{name:10s}: {pos:7.3f} rad ({deg:7.2f}°)")
        print("="*60 + "\n")
        
    def test_basic_movements(self):
        """기본 동작 테스트"""
        self.get_logger().info("\n" + "🎯"*25)
        self.get_logger().info("기본 동작 테스트 시작")
        self.get_logger().info("🎯"*25 + "\n")
        
        # Home
        self.move_arm('home', duration_sec=3.0)
        
        # Ready
        self.move_arm('ready', duration_sec=2.5)
        
        # Left
        self.move_arm('left', duration_sec=2.5)
        
        # Right
        self.move_arm('right', duration_sec=3.0)
        
        # Up
        self.move_arm('up', duration_sec=2.5)
        
        # Home
        self.move_arm('home', duration_sec=3.0)
        
    def test_gripper(self):
        """그리퍼 테스트"""
        self.get_logger().info("\n" + "🤏"*25)
        self.get_logger().info("그리퍼 테스트 시작")
        self.get_logger().info("🤏"*25 + "\n")
        
        for i in range(3):
            self.get_logger().info(f"테스트 {i+1}/3")
            self.move_gripper('open', duration_sec=1.5)
            self.move_gripper('close', duration_sec=1.5)
            
    def test_pickup_place_scenario(self):
        """픽업 & 배치 시나리오"""
        self.get_logger().info("\n" + "🍔"*25)
        self.get_logger().info("픽업 & 배치 시나리오 시작")
        self.get_logger().info("🍔"*25 + "\n")
        
        # 1. Ready
        self.get_logger().info("1️⃣ Ready 자세")
        self.move_arm('ready', duration_sec=3.0)
        
        # 2. 그리퍼 열기
        self.get_logger().info("2️⃣ 그리퍼 열기")
        self.move_gripper('open', duration_sec=1.5)
        
        # 3. Pickup 위치
        self.get_logger().info("3️⃣ Pickup 위치로 접근")
        self.move_arm('pickup', duration_sec=2.5)
        
        # 4. 잡기
        self.get_logger().info("4️⃣ 물체 잡기")
        self.move_gripper('close', duration_sec=1.5)
        time.sleep(0.5)
        
        # 5. 올리기
        self.get_logger().info("5️⃣ 안전 높이로 올리기")
        self.move_arm('up', duration_sec=2.0)
        
        # 6. Place 위치로 이동
        self.get_logger().info("6️⃣ Place 위치로 이동")
        self.move_arm('place', duration_sec=3.0)
        
        # 7. 놓기
        self.get_logger().info("7️⃣ 물체 놓기")
        self.move_gripper('open', duration_sec=1.5)
        time.sleep(0.5)
        
        # 8. 후퇴
        self.get_logger().info("8️⃣ 후퇴")
        self.move_arm('ready', duration_sec=2.5)
        
        # 9. Home
        self.get_logger().info("9️⃣ Home으로 복귀")
        self.move_arm('home', duration_sec=3.0)
        
        self.get_logger().info("\n✅ 시나리오 완료!")

def main():
    rclpy.init()
    robot = IntegratedRobotTest()
    
    try:
        print("\n" + "="*60)
        print("🤖 통합 로봇 테스트")
        print("="*60)
        print("선택하세요:")
        print("1. 기본 동작 테스트 (Home → Ready → Left → Right → Up → Home)")
        print("2. 그리퍼 테스트 (열기/닫기 반복)")
        print("3. 픽업 & 배치 시나리오 (전체 워크플로우)")
        print("4. 전체 테스트 (1 + 2 + 3)")
        print("5. 현재 상태 확인")
        print("="*60)
        
        choice = input("선택 (1-5): ").strip()
        
        if choice == '1':
            robot.test_basic_movements()
        elif choice == '2':
            robot.test_gripper()
        elif choice == '3':
            robot.test_pickup_place_scenario()
        elif choice == '4':
            robot.test_basic_movements()
            robot.test_gripper()
            robot.test_pickup_place_scenario()
        elif choice == '5':
            robot.print_current_state()
        else:
            robot.get_logger().error("❌ 잘못된 선택입니다.")
            
    except KeyboardInterrupt:
        robot.get_logger().info("\n⚠️ 사용자가 중단했습니다.")
    except Exception as e:
        robot.get_logger().error(f"❌ 에러 발생: {e}")
    finally:
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
