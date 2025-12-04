#!/usr/bin/env python3
# safety_monitor_node.py
# [GUI Ver] 웹캠 화면 + 위험 구역 시각화 + 로그 출력

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# =========================================================
# 🎛️ CONFIGURATION
# =========================================================
WEBCAM_INDEX = 0        # 웹캠 번호
CONF_THRESHOLD = 0.4    # 사람 인식 정확도 기준
PIXEL_TOUCH_LIMIT = 10  # 10픽셀 이상 겹치면 감지

# [위험 구역 설정] (화면 오른쪽 절반 예시)
# 초록색/빨간색 네모로 표시될 영역입니다.
DANGER_ZONE_POINTS = np.array([
    [320, 0],    # 상단 중앙
    [640, 0],    # 상단 우측
    [640, 480],  # 하단 우측
    [320, 480]   # 하단 중앙
], np.int32)
# =========================================================

class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor_node')
        
        self.get_logger().info("=========================================")
        self.get_logger().info("👀 Safety Monitor: GUI 디버깅 모드")
        self.get_logger().info("=========================================")

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

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 1. 초기 1회 위험 구역 마스크 생성
        if self.zone_mask is None:
            self.zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(self.zone_mask, [DANGER_ZONE_POINTS], 255)

        # 2. 사람 감지 (YOLO Segmentation)
        results = self.model(frame, classes=0, verbose=False, conf=CONF_THRESHOLD)
        
        current_danger = False
        detected_mask_overlay = np.zeros(frame.shape[:2], dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                for seg in result.masks.data:
                    mask = seg.cpu().numpy().astype(np.uint8) * 255
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    detected_mask_overlay = cv2.bitwise_or(detected_mask_overlay, mask)

        # 3. 겹침 확인 (핵심 로직)
        overlap = cv2.bitwise_and(self.zone_mask, detected_mask_overlay)
        overlap_count = cv2.countNonZero(overlap)

        if overlap_count > PIXEL_TOUCH_LIMIT:
            current_danger = True

        # 4. 로그 출력
        if current_danger:
            if not self.is_danger_state:
                print("\n🚨🚨🚨 영역 내 사람이 들어옴! 🚨🚨🚨\n")
                self.is_danger_state = True
        else:
            if self.is_danger_state:
                print("✅ 사람이 나갔습니다. (안전)")
                self.is_danger_state = False

        # =========================================================
        # 📺 [화면 디버깅] 시각화 코드
        # =========================================================
        
        # (1) 상태에 따른 색상 설정 (위험: 빨강 / 안전: 초록)
        color = (0, 0, 255) if current_danger else (0, 255, 0)
        thickness = 3 if current_danger else 2
        status_text = "DANGER !!!" if current_danger else "SAFE"

        # (2) 위험 구역(Polygon) 그리기
        cv2.polylines(frame, [DANGER_ZONE_POINTS], True, color, thickness)

        # (3) 감지된 사람 빨간색 반투명 표시
        if cv2.countNonZero(detected_mask_overlay) > 0:
            color_overlay = np.zeros_like(frame)
            # 마스크 영역을 빨간색으로 칠함
            color_overlay[:, :, 2] = detected_mask_overlay 
            # 원본과 합성 (투명도 조절)
            frame = cv2.addWeighted(frame, 1.0, color_overlay, 0.5, 0)

        # (4) 텍스트 출력
        cv2.putText(frame, f"Status: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # (5) 윈도우 띄우기
        cv2.imshow("Safety Monitor WebCam", frame)
        cv2.waitKey(1) # 화면 갱신 필수

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows() # 창 닫기
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()