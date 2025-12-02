#!/usr/bin/env python3
# virtual_gripper.py
# 가제보용 가상 그리퍼 인터페이스

import time

class VirtualGripper:
    def __init__(self, moveit_client):
        """
        초기화
        :param moveit_client: MoveItActionClient 인스턴스 (메인 컨트롤러에서 전달받음)
        """
        self.moveit = moveit_client
        self.GROUP_NAME = "gripper"
        self.JOINT_NAME = ["rg2_finger_joint1"] # 제어할 메인 관절

        # 그리퍼 상태 값 (라디안)
        self.OPEN_VAL = 0.0
        self.CLOSE_VAL = 0.6  # 닫혔을 때 각도 (물체 크기에 따라 조절 가능)

    def open_gripper(self):
        """그리퍼 벌리기"""
        # print("👐 가상 그리퍼: 열기")
        return self.moveit.move_to_joints(self.GROUP_NAME, self.JOINT_NAME, [self.OPEN_VAL])

    def close_gripper(self):
        """그리퍼 오므리기"""
        # print("✊ 가상 그리퍼: 닫기")
        return self.moveit.move_to_joints(self.GROUP_NAME, self.JOINT_NAME, [self.CLOSE_VAL])

    # (선택) 실제 로봇 코드와의 호환성을 위한 더미 함수들
    def open_connection(self): pass
    def close_connection(self): pass