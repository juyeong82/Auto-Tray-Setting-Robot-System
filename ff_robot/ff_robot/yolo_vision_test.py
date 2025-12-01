#!/usr/bin/env python3
# yolo_vision_node_debug.py
# [Debug] YOLO 결과 객체 내부를 뜯어서 OBB 존재 여부 확인

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import math

from ff_robot_interfaces.srv import DetectObject 
from geometry_msgs.msg import Point

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        self.get_logger().info("=========================================")
        self.get_logger().info("🚀 YOLO Vision Node (DEBUG MODE)")
        self.get_logger().info("=========================================")
        
        # 모델 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "models", "withobb.pt")
        self.bridge = CvBridge()
        self.latest_color_img = None
        
        try:
            self.model = YOLO(self.model_path) 
            self.get_logger().info(f"✅ 모델 로드: {self.model_path}")
            # [중요] 모델의 Task 타입 확인
            self.get_logger().info(f"ℹ️  모델 Task 타입: {self.model.task}") 
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self.cb_group = ReentrantCallbackGroup()
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, qos_profile, callback_group=self.cb_group)
        self.srv = self.create_service(DetectObject, '/dsr01/detect_object', self.handle_detect_object, callback_group=self.cb_group)
        self.get_logger().info("👀 Debug Service Ready")

    def color_callback(self, msg):
        try: self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def handle_detect_object(self, request, response):
        target_name = request.target_object_id
        self.get_logger().info(f"\n🔎 [DEBUG] 요청: '{target_name}'")

        if self.latest_color_img is None:
            self.get_logger().error("이미지 없음")
            response.found = False
            return response

        # 추론 실행
        results = self.model(self.latest_color_img, verbose=False, conf=0.25)
        
        # [핵심] 결과 객체 내부 조사
        for i, r in enumerate(results):
            self.get_logger().info(f"--- Result {i} ---")
            
            # 1. OBB 속성 확인
            if r.obb is not None:
                self.get_logger().info(f"✅ [OBB 감지됨!] 개수: {len(r.obb)}")
                for box in r.obb:
                    # xywhr: x, y, w, h, rotation(radian)
                    data = box.xywhr[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    self.get_logger().info(f"   - Class: {name}")
                    self.get_logger().info(f"   - Raw Data (xywhr): {data}")
                    self.get_logger().info(f"   - 회전값(deg): {math.degrees(data[4]):.2f}")
            else:
                self.get_logger().warn("❌ [OBB 속성 없음] r.obb is None")

            # 2. 일반 Box 속성 확인
            if r.boxes is not None:
                self.get_logger().info(f"📦 [일반 Box 감지됨] 개수: {len(r.boxes)}")
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    self.get_logger().info(f"   - Class: {name} (일반 박스)")
            else:
                self.get_logger().info("   [일반 Box 속성 없음]")

        response.found = False # 디버깅만 하고 종료
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()