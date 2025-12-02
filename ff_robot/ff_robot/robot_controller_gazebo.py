#!/usr/bin/env python3
# robot_controller_gazebo.py
# 라이브러리 없이 순수 ROS 2 Action으로 MoveIt 제어 (Non-blocking 초기화 적용)

import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
from scipy.spatial.transform import Rotation as R

# MoveIt 2 Action & Messages
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint
from moveit_msgs.msg import MotionPlanRequest
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from shape_msgs.msg import SolidPrimitive

# 커스텀 인터페이스
# [수정] OrderService -> OrderService2 로 변경 (키오스크와 통일)
from ff_robot_interfaces.srv import OrderService2, DetectObject
from ff_robot.order_logic import SlotManager
from ff_robot.tray_manager_gazebo import TrayManager
from ff_robot.virtual_gripper import VirtualGripper

# =========================================================
# [🦾 순수 ROS 2 Action 기반 MoveIt 클라이언트]
# =========================================================
class MoveItActionClient:
    def __init__(self, node: Node):
        self.node = node
        self._action_client = ActionClient(node, MoveGroup, 'move_action', callback_group=ReentrantCallbackGroup())
        
        # 생성자에서는 1초만 대기하고 넘어감
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().warn("⚠️ MoveGroup 서버가 아직 준비되지 않았습니다. 이동 명령 시 다시 확인합니다.")

    def ensure_connected(self):
        """이동 명령 직전에 연결 확인"""
        if not self._action_client.server_is_ready():
            self.node.get_logger().info("⏳ MoveGroup 서버 연결 대기 중...")
            self._action_client.wait_for_server()
            self.node.get_logger().info("✅ MoveGroup 서버 연결됨!")

    def _send_goal(self, request_msg):
        self.ensure_connected() # 연결 확인
        
        goal_msg = MoveGroup.Goal()
        goal_msg.request = request_msg
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        
        # 동기 대기 (While Loop)
        while not send_goal_future.done():
            time.sleep(0.01)
            
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('❌ 경로 계획 거부됨')
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
            
        result = result_future.result().result
        if result.error_code.val == 1: # SUCCESS
            return True
        else:
            self.node.get_logger().error(f'❌ 이동 실패 (에러 코드: {result.error_code.val})')
            return False

    def move_to_joints(self, group_name, joint_names, target_values):
        req = MotionPlanRequest()
        req.group_name = group_name
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.allowed_planning_time = 5.0
        
        constraints = Constraints()
        for name, val in zip(joint_names, target_values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        req.goal_constraints.append(constraints)
        return self._send_goal(req)

    def move_to_pose(self, group_name, end_effector_link, x, y, z, rx, ry, rz):
        req = MotionPlanRequest()
        req.group_name = group_name
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.allowed_planning_time = 5.0
        req.num_planning_attempts = 10
        
        pc = PositionConstraint()
        pc.header.frame_id = "base_link"
        pc.link_name = end_effector_link
        pc.weight = 1.0
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.001, 0.001, 0.001]
        pc.constraint_region.primitives.append(box)
        
        target_point = Point(x=x/1000.0, y=y/1000.0, z=z/1000.0)
        pose = PoseStamped()
        pose.pose.position = target_point
        pose.pose.orientation.w = 1.0 
        pc.constraint_region.primitive_poses.append(pose.pose)
        
        oc = OrientationConstraint()
        oc.header.frame_id = "base_link"
        oc.link_name = end_effector_link
        oc.weight = 1.0
        
        q = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_quat()
        oc.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        
        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        req.goal_constraints.append(constraints)
        
        return self._send_goal(req)

# =========================================================
# [📡 메인 노드 로직]
# =========================================================
class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller_node')
        
        self.slot_manager = SlotManager()
        self.tray_manager = TrayManager()
        
        self.moveit = MoveItActionClient(self)
        self.gripper = VirtualGripper(self.moveit)
        
        self.ARM_GROUP = "manipulator"
        self.EE_LINK = 'link_6'
        self.ARM_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        self.cb_group = ReentrantCallbackGroup()
        
        # [수정] 서비스 타입을 OrderService2로 변경
        self.create_service(OrderService2, '/dsr01/order_service', self.handle_order, callback_group=self.cb_group)
        self.vision_cli = self.create_client(DetectObject, '/dsr01/detect_object', callback_group=self.cb_group)
        
        self.get_logger().info("✅ [Virtual] 로봇 컨트롤러 노드 시작 (Spinning...)")
        
        # 생성자에서 바로 움직이지 않고, 1초 뒤에 타이머로 실행 (Non-blocking)
        self.create_timer(1.0, self.initial_homing_callback)
        self.is_homed = False

    def initial_homing_callback(self):
        if not self.is_homed:
            self.get_logger().info("🏠 초기화: 홈 위치로 이동 시도...")
            self.go_home()
            self.is_homed = True
            self.get_logger().info("✨ 로봇 초기화 완료! 주문 대기 중.")

    def go_home(self):
        home_joints = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
        self.moveit.move_to_joints(self.ARM_GROUP, self.ARM_JOINTS, home_joints)

    def handle_order(self, request, response):
        import json
        self.get_logger().info(f"🧾 주문 수신: {request.order_data_json}")
        try:
            order_data = json.loads(request.order_data_json)
            
            for item_name, qty in order_data.items():
                for _ in range(qty):
                    if not self.process_item(item_name):
                        response.success = False
                        response.message = f"{item_name} 처리 실패"
                        return response
                        
            response.success = True
            response.message = "주문 처리 완료"
            self.go_home() # 작업 끝나면 홈으로
            
        except Exception as e:
            self.get_logger().error(f"주문 처리 중 오류: {e}")
            response.success = False
            response.message = str(e)
            
        return response

    def process_item(self, item_name):
        self.get_logger().info(f"🦾 아이템 처리 시작: {item_name}")
        
        if item_name not in self.tray_manager.PICKUP_ZONES:
            self.get_logger().error(f"❌ 위치 정보 없음: {item_name}")
            return False
            
        pick = self.tray_manager.PICKUP_ZONES[item_name]
        
        # 1. Pick
        self.get_logger().info(f"   📍 픽업 이동: {pick}")
        self.moveit.move_to_pose(self.ARM_GROUP, self.EE_LINK, pick[0], pick[1], pick[2]+150, pick[3], pick[4], pick[5])
        self.gripper.open_gripper()
        self.moveit.move_to_pose(self.ARM_GROUP, self.EE_LINK, pick[0], pick[1], pick[2], pick[3], pick[4], pick[5])
        self.gripper.close_gripper()
        self.moveit.move_to_pose(self.ARM_GROUP, self.EE_LINK, pick[0], pick[1], pick[2]+200, pick[3], pick[4], pick[5])
        
        # 2. Place Calc
        slot_id = self.slot_manager._find_empty_slot() or 1
        tray_center = self.tray_manager.WORK_TABLE_POS
        offset = self.slot_manager.TRAY_OFFSETS[slot_id-1]
        
        place_x = tray_center[0] + offset[0]
        place_y = tray_center[1] + offset[1]
        place_z = tray_center[2] + offset[2] + 80 # 높이 보정
        
        # 3. Place
        self.get_logger().info(f"   📍 배치 이동: 슬롯 {slot_id}")
        self.moveit.move_to_pose(self.ARM_GROUP, self.EE_LINK, place_x, place_y, place_z+150, 0, 180, 90)
        self.moveit.move_to_pose(self.ARM_GROUP, self.EE_LINK, place_x, place_y, place_z, 0, 180, 90)
        self.gripper.open_gripper()
        self.moveit.move_to_pose(self.ARM_GROUP, self.EE_LINK, place_x, place_y, place_z+150, 0, 180, 90)
        
        return True

def main():
    rclpy.init()
    node = RobotControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()