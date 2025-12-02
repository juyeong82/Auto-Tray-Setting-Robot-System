import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive

# 쿼터니언 변환용 (Euler -> Quaternion)
def euler_to_quaternion(roll, pitch, yaw):
    import math
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

class MoveItActionClient(Node):
    def __init__(self):
        super().__init__('moveit_action_client')
        # MoveGroup 액션 클라이언트 생성
        self._action_client = ActionClient(self, MoveGroup, 'move_group')

    def send_goal(self, target_pose):
        self.get_logger().info('MoveIt 액션 서버 연결 대기 중...')
        self._action_client.wait_for_server()

        # --- MoveIt 목표 메시지 구성 ---
        goal_msg = MoveGroup.Goal()
        
        # 1. 설정: 어떤 그룹을 움직일 것인가?
        goal_msg.request.group_name = 'dsr01'  # 두산 로봇 MoveIt Config의 그룹 이름 (보통 arm, dsr, dsr01 등 확인 필요)
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.5  # 속도 50% 제한 (안전)
        goal_msg.request.max_acceleration_scaling_factor = 0.5

        # 2. 목표 제약 조건 생성 (Position + Orientation)
        # 2-1. 위치 제약
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "link6"  # 로봇 끝단 링크 이름 (보통 link6 또는 tool0)
        
        # 목표 범위 (박스 형태) 설정 - 작게 할수록 정밀
        s = SolidPrimitive()
        s.type = SolidPrimitive.BOX
        s.dimensions = [0.01, 0.01, 0.01] # 1cm 오차 허용
        pos_constraint.constraint_region.primitives.append(s)
        
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0

        # 2-2. 회전 제약
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "base_link"
        ori_constraint.link_name = "link6"
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        # 3. 제약 조건 합치기
        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        goal_msg.request.goal_constraints.append(constraints)

        # 4. 계획 옵션
        # 'plan_only = False'로 설정하면 계획 후 즉시 실행함
        goal_msg.planning_options.plan_only = False 
        goal_msg.planning_options.look_around = True # 장애물 회피 활성화
        goal_msg.planning_options.replan = True      # 실패 시 재계획

        self.get_logger().info('목표 전송 중... (장애물 회피 경로 계산)')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt이 목표를 거부했습니다.')
            return

        self.get_logger().info('경로 계획 성공! 이동 시작...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        # error_code 1 이면 성공 (MoveItErrorCodes)
        if result.error_code.val == 1:
            self.get_logger().info('✅ 이동 완료!')
        else:
            self.get_logger().error(f'❌ 이동 실패 (에러 코드: {result.error_code.val})')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = MoveItActionClient()

    # --- [목표 좌표 설정] ---
    # 로봇 베이스(base_link) 기준 좌표
    target = PoseStamped()
    target.header.frame_id = "base_link"
    
    # 1. 위치 (단위: m) - 장애물 너머의 목표 지점
    # 예: X=0.5m, Y=0.0m, Z=0.4m
    target.pose.position.x = 0.5
    target.pose.position.y = 0.0
    target.pose.position.z = 0.4 

    # 2. 회전 (쿼터니언) - 그리퍼가 아래를 보도록
    # (Roll=180도, Pitch=0, Yaw=90도 등 로봇에 맞게 설정 필요)
    # 아래는 예시 값입니다. 현재 로봇 자세를 참고하세요.
    q = euler_to_quaternion(3.14, 0, 1.57) 
    target.pose.orientation = q

    # 실행
    action_client.send_goal(target)
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()