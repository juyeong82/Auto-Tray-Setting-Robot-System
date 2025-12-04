#!/usr/bin/env python3
# safety_monitor_node.py
# [Final] 웹캠 감시 -> /robot_speed 토픽 발행 -> 로봇 감속

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32  # [추가] 토픽 메시지 타입
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# =========================================================
# 🎛️ CONFIGURATION
# =========================================================
WEBCAM_INDEX = 8
CONF_THRESHOLD = 0.4    
PIXEL_TOUCH_LIMIT = 10  

# [위험 구역 설정]
DANGER_ZONE_POINTS = np.array([
    [0, 0],    # 상단 좌측
    [320, 0],    # 상단 우측
    [320, 480],  # 하단 우측
    [0, 480]   # 하단 좌측
], np.int32)
# =========================================================

class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor_node')
        
        self.get_logger().info("=========================================")
        self.get_logger().info("🛡️ Safety Monitor: Active Mode")
        self.get_logger().info("=========================================")

        # [핵심] 속도 제어 퍼블리셔 생성
        self.speed_pub = self.create_publisher(Int32, '/robot_speed', 10)

        # YOLO 모델 로드
        try:
            self.model = YOLO('yolov8n-seg.pt')
            self.get_logger().info("✅ YOLOv8 모델 준비 완료")
        except Exception as e:
            self.get_logger().error(f"❌ 모델 로드 실패: {e}")
            sys.exit(1)

        # 웹캠 연결
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ 웹캠({WEBCAM_INDEX})을 열 수 없습니다.")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 상태 관리
        self.is_danger_state = False
        self.timer = self.create_timer(0.1, self.timer_callback) # 0.1초 간격

        self.zone_mask = None

    def publish_speed_command(self, speed):
        """속도 명령 전송 함수"""
        msg = Int32()
        msg.data = speed
        self.speed_pub.publish(msg)
        # 로그 출력 (송신 확인용)
        status = "SLOW (30%)" if speed == 30 else "NORMAL (100%)"
        self.get_logger().info(f"📤 [Signal] 속도 명령 전송: {status}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 1. 초기 1회 위험 구역 마스크 생성
        if self.zone_mask is None:
            self.zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(self.zone_mask, [DANGER_ZONE_POINTS], 255)

        # 2. 사람 감지
        results = self.model(frame, classes=0, verbose=False, conf=CONF_THRESHOLD)
        
        current_danger = False
        detected_mask_overlay = np.zeros(frame.shape[:2], dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                for seg in result.masks.data:
                    mask = seg.cpu().numpy().astype(np.uint8) * 255
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    detected_mask_overlay = cv2.bitwise_or(detected_mask_overlay, mask)

        # 3. 겹침 확인
        overlap = cv2.bitwise_and(self.zone_mask, detected_mask_overlay)
        overlap_count = cv2.countNonZero(overlap)

        if overlap_count > PIXEL_TOUCH_LIMIT:
            current_danger = True

        # 4. 상태 변경 시 토픽 발행 및 로그 출력
        if current_danger:
            if not self.is_danger_state:
                print("\n🚨🚨🚨 [DANGER] 작업자 접근! -> 감속 명령(10%) 전송 🚨🚨🚨\n")
                self.publish_speed_command(30) # 30%로 감속
                self.is_danger_state = True
        else:
            if self.is_danger_state:
                print("\n✅ [SAFE] 안전 구역 확보 -> 정상 속도(100%) 복귀\n")
                self.publish_speed_command(100) # 100%로 복귀
                self.is_danger_state = False

        # =========================================================
        # 📺 [화면 디버깅] 시각화 코드
        # =========================================================
        color = (0, 0, 255) if current_danger else (0, 255, 0)
        thickness = 3 if current_danger else 2
        status_text = "DANGER (SLOW)" if current_danger else "SAFE (NORMAL)"

        cv2.polylines(frame, [DANGER_ZONE_POINTS], True, color, thickness)

        if cv2.countNonZero(detected_mask_overlay) > 0:
            color_overlay = np.zeros_like(frame)
            color_overlay[:, :, 2] = detected_mask_overlay 
            frame = cv2.addWeighted(frame, 1.0, color_overlay, 0.5, 0)

        cv2.putText(frame, f"Status: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Safety Monitor WebCam", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()