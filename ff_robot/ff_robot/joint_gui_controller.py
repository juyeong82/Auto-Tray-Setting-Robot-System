#!/usr/bin/env python3
"""
간단한 GUI 조인트 컨트롤러
tkinter로 슬라이더를 만들어 로봇을 실시간으로 제어합니다.
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import tkinter as tk
from tkinter import ttk
import threading
import math

class JointControllerGUI(Node):
    def __init__(self):
        super().__init__('joint_controller_gui')
        
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
        
        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_positions = {
            'joint_1': 0.0,
            'joint_2': 0.0,
            'joint_3': 0.0,
            'joint_4': 0.0,
            'joint_5': 0.0,
            'joint_6': 0.0,
            'rg2_finger_joint1': 0.0
        }
        
        self.get_logger().info("🎮 GUI 조인트 컨트롤러 시작!")
        
    def joint_state_callback(self, msg):
        """현재 조인트 상태 업데이트"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]
                
    def move_arm_joint(self, joint_values):
        """팔 조인트 이동"""
        msg = JointTrajectory()
        msg.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = joint_values
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5초
        msg.points.append(point)
        
        self.arm_pub.publish(msg)
        
    def move_gripper(self, position):
        """그리퍼 이동"""
        msg = JointTrajectory()
        msg.joint_names = ['rg2_finger_joint1']
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=0, nanosec=500000000)
        msg.points.append(point)
        
        self.gripper_pub.publish(msg)

class ControllerWindow:
    def __init__(self, controller_node):
        self.node = controller_node
        
        # 메인 윈도우
        self.root = tk.Tk()
        self.root.title("🤖 로봇 조인트 컨트롤러")
        self.root.geometry("600x750")
        self.root.configure(bg='#2b2b2b')
        
        # 조인트 슬라이더
        self.sliders = {}
        self.labels = {}
        
        # 조인트 정보 (이름, 최소값, 최대값)
        joints = [
            ('joint_1', -3.14, 3.14),
            ('joint_2', -3.14, 3.14),
            ('joint_3', -3.14, 3.14),
            ('joint_4', -3.14, 3.14),
            ('joint_5', -3.14, 3.14),
            ('joint_6', -3.14, 3.14),
        ]
        
        # 타이틀
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(pady=10)
        
        title = tk.Label(
            title_frame, 
            text="🎮 조인트 컨트롤러",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        title.pack()
        
        # 메인 프레임
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 팔 조인트 슬라이더
        arm_label = tk.Label(
            main_frame,
            text="━━━ 팔 조인트 ━━━",
            font=('Arial', 12, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        arm_label.pack(pady=(0, 10))
        
        for joint_name, min_val, max_val in joints:
            self.create_slider(main_frame, joint_name, min_val, max_val, 'arm')
            
        # 그리퍼 슬라이더
        gripper_label = tk.Label(
            main_frame,
            text="━━━ 그리퍼 ━━━",
            font=('Arial', 12, 'bold'),
            bg='#2b2b2b',
            fg='#00aaff'
        )
        gripper_label.pack(pady=(20, 10))
        
        self.create_slider(main_frame, 'rg2_finger_joint1', 0.0, 0.055, 'gripper')
        
        # 버튼 프레임
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=20)
        
        # Home 버튼
        home_btn = tk.Button(
            button_frame,
            text="🏠 Home",
            command=self.go_home,
            font=('Arial', 11, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            relief=tk.RAISED,
            cursor='hand2'
        )
        home_btn.pack(side=tk.LEFT, padx=5)
        
        # 그리퍼 열기 버튼
        open_btn = tk.Button(
            button_frame,
            text="✋ 열기",
            command=self.open_gripper,
            font=('Arial', 11, 'bold'),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            relief=tk.RAISED,
            cursor='hand2'
        )
        open_btn.pack(side=tk.LEFT, padx=5)
        
        # 그리퍼 닫기 버튼
        close_btn = tk.Button(
            button_frame,
            text="✊ 닫기",
            command=self.close_gripper,
            font=('Arial', 11, 'bold'),
            bg='#FF5722',
            fg='white',
            padx=20,
            pady=10,
            relief=tk.RAISED,
            cursor='hand2'
        )
        close_btn.pack(side=tk.LEFT, padx=5)
        
        # 타이머로 현재 상태 업데이트
        self.update_current_positions()
        
    def create_slider(self, parent, joint_name, min_val, max_val, joint_type):
        """슬라이더 생성"""
        frame = tk.Frame(parent, bg='#2b2b2b')
        frame.pack(fill=tk.X, pady=5)
        
        # 라벨
        label = tk.Label(
            frame,
            text=f"{joint_name}:",
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='white',
            width=20,
            anchor='w'
        )
        label.pack(side=tk.LEFT)
        
        # 값 표시
        value_label = tk.Label(
            frame,
            text="0.000",
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='#ffaa00',
            width=10
        )
        value_label.pack(side=tk.RIGHT)
        self.labels[joint_name] = value_label
        
        # 슬라이더
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            resolution=0.001,
            length=300,
            bg='#3b3b3b',
            fg='white',
            highlightbackground='#2b2b2b',
            troughcolor='#555555',
            command=lambda val, jn=joint_name, jt=joint_type: self.on_slider_change(jn, float(val), jt)
        )
        slider.set(0.0)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        self.sliders[joint_name] = slider
        
    def on_slider_change(self, joint_name, value, joint_type):
        """슬라이더 값 변경 시"""
        # 라벨 업데이트
        if joint_name == 'rg2_finger_joint1':
            self.labels[joint_name].config(text=f"{value:.3f}")
        else:
            deg = math.degrees(value)
            self.labels[joint_name].config(text=f"{value:.3f} ({deg:.1f}°)")
        
        # 로봇 제어
        if joint_type == 'arm':
            joint_values = [
                self.sliders['joint_1'].get(),
                self.sliders['joint_2'].get(),
                self.sliders['joint_3'].get(),
                self.sliders['joint_4'].get(),
                self.sliders['joint_5'].get(),
                self.sliders['joint_6'].get(),
            ]
            self.node.move_arm_joint(joint_values)
        else:
            self.node.move_gripper(value)
            
    def update_current_positions(self):
        """현재 위치 표시 업데이트"""
        # 슬라이더를 현재 위치로 업데이트 (사용자가 움직이지 않을 때만)
        # 여기서는 값만 표시
        self.root.after(100, self.update_current_positions)
        
    def go_home(self):
        """Home 자세"""
        for slider in self.sliders.values():
            slider.set(0.0)
            
    def open_gripper(self):
        """그리퍼 열기"""
        self.sliders['rg2_finger_joint1'].set(0.055)
        
    def close_gripper(self):
        """그리퍼 닫기"""
        self.sliders['rg2_finger_joint1'].set(0.0)
        
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

def spin_node(node):
    """별도 스레드에서 ROS spin"""
    rclpy.spin(node)

def main():
    rclpy.init()
    
    # 노드 생성
    controller = JointControllerGUI()
    
    # ROS spin을 별도 스레드에서 실행
    spin_thread = threading.Thread(target=spin_node, args=(controller,), daemon=True)
    spin_thread.start()
    
    # GUI 실행
    window = ControllerWindow(controller)
    
    try:
        window.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
