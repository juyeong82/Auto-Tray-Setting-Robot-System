#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import PointCloud2
import time

class ObstacleAvoidanceTest(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_test')
        
        # Planning Scene 서비스 클라이언트
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, 
            '/apply_planning_scene'
        )
        
        # PointCloud2 구독 (Octomap 업데이트용)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )
        
        # Planning Scene Publisher (Octomap 반영)
        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        
        self.get_logger().info("✅ Obstacle Avoidance Test Node Ready")
        
    def pointcloud_callback(self, msg):
        # PointCloud2를 Planning Scene에 실시간 반영
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.world.octomap.header = msg.header
        # Octomap은 MoveIt의 OctomapUpdater가 자동 처리
        self.scene_pub.publish(scene_msg)
        
    def add_static_obstacle(self, name, position, size):
        """테스트용 정적 장애물 추가"""
        collision_obj = CollisionObject()
        collision_obj.header.frame_id = "base_0"
        collision_obj.id = name
        
        # Box 형태 장애물
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = size  # [x, y, z]
        
        pose = PoseStamped()
        pose.header.frame_id = "base_0"
        pose.pose.position.x = position[0] / 1000.0  # mm to m
        pose.pose.position.y = position[1] / 1000.0
        pose.pose.position.z = position[2] / 1000.0
        pose.pose.orientation.w = 1.0
        
        collision_obj.primitives.append(box)
        collision_obj.primitive_poses.append(pose.pose)
        collision_obj.operation = CollisionObject.ADD
        
        # Planning Scene에 적용
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.world.collision_objects.append(collision_obj)
        
        while not self.apply_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')
        
        req = ApplyPlanningScene.Request()
        req.scene = scene_msg
        future = self.apply_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().success:
            self.get_logger().info(f"✅ Obstacle '{name}' added")
        else:
            self.get_logger().error(f"❌ Failed to add '{name}'")

def main():
    rclpy.init()
    node = ObstacleAvoidanceTest()
    
    # 테스트용 장애물 추가 (로봇 경로 중간에 박스 배치)
    # 예: 로봇과 목표 지점 사이에 200x200x300mm 박스
    node.add_static_obstacle(
        name="test_obstacle",
        position=[400.0, 0.0, 200.0],  # [x, y, z] in mm
        size=[0.2, 0.2, 0.3]  # [x, y, z] in meters
    )
    
    node.get_logger().info("🚀 장애물 설정 완료. RViz에서 경로 계획 테스트 가능")
    node.get_logger().info("   Planning Request: Start [0,0,90,0,90,0] → Goal [567.36, 7.33, 194.52, 70.74, 180, 70.60]")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
