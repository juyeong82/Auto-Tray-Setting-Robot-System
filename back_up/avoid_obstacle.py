import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, Constraints, PositionConstraint
from moveit_msgs.action import MoveGroup
import time

from moveit_msgs.msg import OrientationConstraint

class EasyAvoidanceNode(Node):
    def __init__(self):
        super().__init__('easy_avoidance_node')
        
        # --- 설정 확인 ---
        self.frame_id = "base_link"       
        self.end_effector_link = "link_6" 
        self.group_name = "manipulator"   
        # ----------------

        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', qos)
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._action_client.wait_for_server()

    def publish_obstacle(self):
        # 1. 기존 장애물 제거 (청소)
        clean_box = CollisionObject()
        clean_box.header.frame_id = self.frame_id
        clean_box.id = "obstacle_box"
        clean_box.operation = CollisionObject.REMOVE
        self.collision_pub.publish(clean_box)
        time.sleep(0.5)

        # 2. 새 장애물 생성 (위치를 옆으로 옮김: y=0.3)
        box = CollisionObject()
        box.header.frame_id = self.frame_id
        box.id = "obstacle_box"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.05, 0.5, 0.3] # 조금 더 얇고 긴 상자(벽)

        box_pose = PoseStamped()
        box_pose.header.frame_id = self.frame_id
        # 위치: 로봇 왼쪽 앞 (x=0.4, y=0.3)
        box_pose.pose.position.x = 0.5
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 0.4
        box_pose.pose.orientation.w = 1.0

        box.primitives.append(primitive)
        box.primitive_poses.append(box_pose.pose)
        box.operation = CollisionObject.ADD

        self.collision_pub.publish(box)
        self.get_logger().info('장애물 생성: 왼쪽(y=0.3)에 배치')

    def send_goal(self):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = self.frame_id
        goal_msg.request.workspace_parameters.min_corner.x = -2.0
        goal_msg.request.workspace_parameters.max_corner.x = 2.0
        goal_msg.request.workspace_parameters.min_corner.y = -2.0
        goal_msg.request.workspace_parameters.max_corner.y = 2.0
        goal_msg.request.workspace_parameters.min_corner.z = -2.0
        goal_msg.request.workspace_parameters.max_corner.z = 2.0

        goal_msg.request.group_name = self.group_name
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 10.0 # 시간 10초로 늘림 (넉넉하게)
        
        goal_msg.request.max_velocity_scaling_factor = 0.2
        goal_msg.request.max_acceleration_scaling_factor = 0.2

        # --- 목표 위치 ---
        # 정면으로 뻗기 (x=0.6, y=0.0)
        # 장애물이 y=0.3에 있으므로 로봇은 안전하게 직진하거나 살짝 오른쪽으로 피해야 함
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.frame_id
        target_pose.pose.position.x = 0.8
        target_pose.pose.position.y = 0.0
        target_pose.pose.position.z = 0.4
        target_pose.pose.orientation.w = 1.0
        
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.frame_id
        pos_constraint.link_name = self.end_effector_link
        
        # 허용 오차 2cm (너무 빡빡하면 실패함)
        pos_constraint.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.02]))
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0

        # [방향 제약 추가]
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.frame_id
        ori_constraint.link_name = self.end_effector_link
        ori_constraint.orientation = target_pose.pose.orientation # 목표 지점의 방향(수평)을 유지해라

        # 허용 오차 (라디안) - 0.1은 약 5.7도
        # 이 값을 너무 작게(0.01) 하면 로봇이 "너무 어려워서 못 가요"라고 할 수 있습니다.
        ori_constraint.absolute_x_axis_tolerance = 0.3 
        ori_constraint.absolute_y_axis_tolerance = 0.3 
        ori_constraint.absolute_z_axis_tolerance = 3.14 # Z축 회전(Yaw)은 자유롭게 허용 (상관없음)

        ori_constraint.weight = 1.0

        # 제약 조건 리스트에 추가 (위치 + 방향)
        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint) # <- 이거 추가

        goal_msg.request.goal_constraints.append(constraints)

        

        self.get_logger().info(f'목표 전송: x={target_pose.pose.position.x}, y={target_pose.pose.position.y}')
        
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('XXX 명령 거부됨 XXX')
            return

        self.get_logger().info('OOO 명령 수락됨. 계산 중...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('!!! 이동 성공 !!!')
        else:
            self.get_logger().error(f'XXX 이동 실패. 코드: {result.error_code.val}')
            
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = EasyAvoidanceNode()
    
    node.publish_obstacle()
    print("RViz에서 장애물이 왼쪽(y=0.3)으로 이동했는지 확인하세요. (3초 대기)")
    time.sleep(3.0)
    
    node.send_goal()
    rclpy.spin(node)

if __name__ == '__main__':
    main()