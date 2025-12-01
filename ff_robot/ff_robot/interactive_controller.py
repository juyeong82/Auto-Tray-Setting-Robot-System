#!/usr/bin/env python3
"""
대화형 로봇 컨트롤러
명령어를 입력하여 실시간으로 로봇을 제어합니다.
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import threading
import math

class InteractiveController(Node):
    def __init__(self):
        super().__init__('interactive_controller')
        
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
        
        # 현재 상태
        self.current_joint_state = None
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.poses = {
            'home': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'ready': [0.0, -0.5, 0.5, 0.0, 0.0, 0.0],
            'left': [0.8, -0.5, 0.5, 0.0, 0.5, 0.0],
            'right': [-0.8, -0.5, 0.5, 0.0, 0.5, 0.0],
            'up': [0.0, -0.3, 0.3, 0.0, 0.8, 0.0],
            'pickup': [0.5, -0.7, 0.8, 0.0, 0.5, 0.0],
            'place': [-0.5, -0.6, 0.7, 0.0, 0.5, 0.0],
        }
        
        self.get_logger().info("✅ 대화형 컨트롤러 준비 완료!")
        
    def joint_state_callback(self, msg):
        self.current_joint_state = msg
        
    def move_arm(self, positions, duration=3.0):
        msg = JointTrajectory()
        msg.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration), nanosec=0)
        msg.points.append(point)
        
        self.arm_pub.publish(msg)
        print(f"✅ 명령 전송: {positions}")
        
    def move_gripper(self, position, duration=1.5):
        msg = JointTrajectory()
        msg.joint_names = ['rg2_finger_joint1']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=int(duration), nanosec=0)
        msg.points.append(point)
        
        self.gripper_pub.publish(msg)
        status = "열림" if position > 0.02 else "닫힘"
        print(f"✅ 그리퍼 {status}")
        
    def print_current_state(self):
        if self.current_joint_state is None:
            print("⚠️ 조인트 상태를 아직 받지 못했습니다.")
            return
            
        print("\n" + "="*60)
        print("📊 현재 조인트 상태")
        print("="*60)
        
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        for name in joint_names:
            try:
                idx = self.current_joint_state.name.index(name)
                pos = self.current_joint_state.position[idx]
                deg = math.degrees(pos)
                print(f"{name:10s}: {pos:7.3f} rad ({deg:7.2f}°)")
            except ValueError:
                print(f"{name:10s}: N/A")
        
        print("="*60 + "\n")
        
    def print_help(self):
        print("\n" + "="*60)
        print("📖 명령어 도움말")
        print("="*60)
        print("자세 명령:")
        for name in self.poses.keys():
            print(f"  {name:12s} - {self.poses[name]}")
        print("\n그리퍼 명령:")
        print("  open         - 그리퍼 열기")
        print("  close        - 그리퍼 닫기")
        print("\n기타 명령:")
        print("  status       - 현재 상태 확인")
        print("  help         - 도움말 표시")
        print("  quit/exit    - 종료")
        print("="*60 + "\n")
        
    def run_interactive(self):
        print("\n" + "🤖"*30)
        print("대화형 로봇 컨트롤러")
        print("🤖"*30)
        self.print_help()
        
        while rclpy.ok():
            try:
                cmd = input("명령 입력 >>> ").strip().lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    print("👋 종료합니다...")
                    break
                elif cmd == 'help' or cmd == '?':
                    self.print_help()
                elif cmd == 'status':
                    self.print_current_state()
                elif cmd == 'open':
                    self.move_gripper(0.055)
                elif cmd == 'close':
                    self.move_gripper(0.0)
                elif cmd in self.poses:
                    self.move_arm(self.poses[cmd])
                elif cmd == '':
                    continue
                else:
                    print(f"❌ 알 수 없는 명령: '{cmd}'. 'help'를 입력하여 도움말을 확인하세요.")
                    
            except KeyboardInterrupt:
                print("\n👋 종료합니다...")
                break
            except Exception as e:
                print(f"❌ 에러: {e}")

def spin_node(node):
    """별도 스레드에서 노드 spin"""
    rclpy.spin(node)

def main():
    rclpy.init()
    controller = InteractiveController()
    
    # ROS spin을 별도 스레드에서 실행
    spin_thread = threading.Thread(target=spin_node, args=(controller,), daemon=True)
    spin_thread.start()
    
    try:
        controller.run_interactive()
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
