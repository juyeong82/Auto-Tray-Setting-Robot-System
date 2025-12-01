#!/usr/bin/env python3
# tray_manager.py
# 트레이 픽업, 배치, 서빙 로직

import numpy as np
from scipy.spatial.transform import Rotation as R

class TrayManager:
    """
    트레이 2개를 관리하는 클래스
    - 트레이 인식 및 우선순위 결정 (X축 거리 기준)
    - 트레이 그립 포인트 계산 (끄트머리 잡기)
    - 배치 위치 계산
    """
    
    def __init__(self):
        # ========================================
        # [🎛️ 튜닝 섹션] 실제 측정 후 수정하세요!
        # ========================================
        
        # 트레이 크기 (단위: mm)
        self.TRAY_LENGTH = 300.0  # X축 방향 (긴 쪽)
        self.TRAY_WIDTH = 200.0   # Y축 방향 (짧은 쪽)
        
        # 트레이 끄트머리 오프셋 (중심에서 끝까지 거리)
        # OBB 중심점에서 이만큼 이동하면 끄트머리가 됨
        self.TRAY_GRIP_OFFSET_X = self.TRAY_LENGTH / 2.0  # 150mm
        
        # 작업 테이블 위치 (로봇 베이스 기준, 단위: mm)
        # [실측 완료] 첫 번째 트레이 배치 위치
        self.WORK_TABLE_POS = [205.0, 20.0, 25.0]  # [X, Y, Z]
        
        # [실측 완료] 작업 테이블 기본 자세 (Roll, Pitch, Yaw)
        # 특이점 회피를 위해 실측값 사용!
        # 슬롯1 실측값을 기본으로 사용
        self.WORK_TABLE_ORIENTATION = [43.35, -180.0, -134.82]  # [Rx, Ry, Rz] degree
        
        # [옵션] 슬롯별 개별 자세 사용 (더 안전)
        # 슬롯마다 자세가 다르면 각각 설정
        self.USE_INDIVIDUAL_SLOT_ORIENTATION = True  # True/False
        self.SLOT1_ORIENTATION = [43.35, -180.0, -134.82]   # 슬롯1 실측
        self.SLOT2_ORIENTATION = [35.09, 179.78, 141.93]     # 슬롯2 실측
        
        # [참고] 첫 번째 트레이 배치 위치 - 조인트 (degree)
        # J_WORK_SLOT1 = [3.63, -18.40, 132.22, 0.03, 66.18, 185.59]
        
        # [참고] 첫 번째 트레이 배치 위치 - 태스크 (mm/degree)
        # X_WORK_SLOT1 = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
        
        # [참고] 두 번째 트레이 배치 위치 - 조인트 (degree)
        # J_WORK_SLOT2 = [2.03, 15.90, 99.42, 0.08, 64.48, 184.20]
        
        # [참고] 두 번째 트레이 배치 위치 - 태스크 (mm/degree)
        # X_WORK_SLOT2 = [244.44, 23.78, 21.53, 35.09, 179.78, 141.93]
        
        # 작업 테이블 회전값 모드
        # "FIXED": 항상 실측 자세 사용 (안전, 권장)
        # "ALIGN": 트레이 회전값(Rz)에 맞춰서 조정 (실험적, 위험!)
        self.WORK_TABLE_RZ_MODE = "FIXED"  # "FIXED" or "ALIGN"
        
        # 트레이 2개 배치 간격 (X축 방향, 단위: mm)
        # 실측: 슬롯1 X=205.0, 슬롯2 X=244.44 → 간격 = 39.44mm
        self.TRAY_SPACING_X = 39.44  # X축 방향 트레이 간 거리
        
        # 서빙 테이블 위치 (손님 측)
        # [실측 완료] 작업 테이블에서 Y축 +260mm 이동
        self.SERVE_TABLE_OFFSET_Y = 260.0  # Y축 방향 오프셋 (mm)
        
        # 서빙 테이블 자세는 작업 테이블과 동일하게 유지
        # (각 슬롯의 실측 자세를 그대로 사용)
        
        # 트레이 상태
        self.tray_states = {
            0: {"status": "empty", "order_id": None},  # 슬롯 0
            1: {"status": "empty", "order_id": None}   # 슬롯 1
        }

    def calculate_tray_grip_point(self, tray_center_pos, tray_rotation_rz):
        """
        트레이 중심점과 회전값으로부터 끄트머리 그립 포인트 계산
        
        Args:
            tray_center_pos: [x, y, z] OBB 중심점 (로봇 베이스 좌표계, mm)
            tray_rotation_rz: OBB 회전값 (degree, Yaw만)
            
        Returns:
            grip_pos: [x, y, z, rx, ry, rz] 그립 포인트 절대 좌표
        """
        cx, cy, cz = tray_center_pos
        
        # 1. 회전 행렬 생성 (Z축 회전만, Yaw)
        rot_rad = np.radians(tray_rotation_rz)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        
        # 2. 로컬 오프셋 (트레이 좌표계에서 끄트머리 방향)
        # X축 양의 방향으로 TRAY_GRIP_OFFSET_X만큼 이동
        local_offset = np.array([self.TRAY_GRIP_OFFSET_X, 0.0, 0.0])
        
        # 3. 회전 적용 (글로벌 좌표계로 변환)
        global_offset_x = local_offset[0] * cos_r - local_offset[1] * sin_r
        global_offset_y = local_offset[0] * sin_r + local_offset[1] * cos_r
        
        # 4. 최종 그립 포인트
        grip_x = cx + global_offset_x
        grip_y = cy + global_offset_y
        grip_z = cz  # Z는 동일
        
        # 5. 그리퍼 회전값 (트레이와 정렬)
        grip_rx = 0.0
        grip_ry = 180.0  # 아래 방향
        grip_rz = tray_rotation_rz  # 트레이 회전과 동일
        
        return [grip_x, grip_y, grip_z, grip_rx, grip_ry, grip_rz]

    def select_nearest_tray(self, detected_trays):
        """
        여러 개의 트레이 중 X축 기준 가장 가까운 것 선택
        
        Args:
            detected_trays: [
                {"position": [x, y, z], "rotation_rz": deg, "confidence": 0.9},
                ...
            ]
            
        Returns:
            nearest_tray: 가장 가까운 트레이 딕셔너리
        """
        if not detected_trays:
            return None
            
        # X축 값 기준 정렬 (작을수록 가까움)
        sorted_trays = sorted(detected_trays, key=lambda t: t["position"][0])
        
        return sorted_trays[0]

    def calculate_work_table_position(self, slot_id, tray_rotation_rz=0.0):
        """
        작업 테이블의 슬롯별 배치 위치 계산
        
        Args:
            slot_id: 1 또는 2 (슬롯 번호)
            tray_rotation_rz: 트레이 회전값 (degree, 기본값 0.0)
            
        Returns:
            position: [x, y, z, rx, ry, rz]
        """
        base_x, base_y, base_z = self.WORK_TABLE_POS
        
        # X축 방향으로 나란히 배치
        # 슬롯1: X=205.0 (기준), 슬롯2: X=244.44 (+39.44)
        offset_x = (slot_id - 1) * self.TRAY_SPACING_X
        
        # 슬롯별 개별 자세 사용 여부
        if self.USE_INDIVIDUAL_SLOT_ORIENTATION:
            # 각 슬롯의 실측 자세 사용 (가장 안전!)
            if slot_id == 1:
                base_rx, base_ry, base_rz = self.SLOT1_ORIENTATION
            else:  # slot_id == 2
                base_rx, base_ry, base_rz = self.SLOT2_ORIENTATION
        else:
            # 공통 자세 사용
            base_rx, base_ry, base_rz = self.WORK_TABLE_ORIENTATION
        
        # 회전값 결정 모드
        if self.WORK_TABLE_RZ_MODE == "FIXED":
            # 실측 자세 그대로 사용 (안전, 권장)
            rx, ry, rz = base_rx, base_ry, base_rz
        elif self.WORK_TABLE_RZ_MODE == "ALIGN":
            # Rz만 트레이 회전값에 맞춤 (실험적, 위험!)
            rx, ry = base_rx, base_ry
            rz = base_rz + tray_rotation_rz  # 오프셋 추가
        else:
            # 기본값
            rx, ry, rz = base_rx, base_ry, base_rz
        
        return [
            base_x + offset_x,  # X축 방향으로 이동
            base_y,             # Y는 동일 (평행)
            base_z,
            rx,   # 실측 Roll (특이점 회피!)
            ry,   # 실측 Pitch
            rz    # 실측 Yaw (또는 조정됨)
        ]

    def calculate_serve_position(self, slot_id=1, work_slot_id=1):
        """
        서빙 테이블 위치 계산 (완료된 트레이를 옮기는 위치)
        작업 테이블에서 Y축 +260mm 이동한 위치
        
        Args:
            slot_id: 서빙 슬롯 번호 (현재는 사용 안 함, 향후 확장용)
            work_slot_id: 어느 작업 슬롯에서 가져올지 (1 or 2)
        
        Returns:
            position: [x, y, z, rx, ry, rz]
        """
        # 작업 테이블 위치 가져오기
        work_pos = self.calculate_work_table_position(work_slot_id, tray_rotation_rz=0.0)
        
        # Y축 +260mm 이동
        return [
            work_pos[0],                              # X: 작업 테이블과 동일
            work_pos[1] + self.SERVE_TABLE_OFFSET_Y,  # Y: +260mm
            work_pos[2],                              # Z: 작업 테이블과 동일
            work_pos[3],                              # Rx: 작업 테이블과 동일
            work_pos[4],                              # Ry: 작업 테이블과 동일
            work_pos[5]                               # Rz: 작업 테이블과 동일
        ]


    def update_tray_status(self, slot_id, status, order_id=None):
        """트레이 상태 업데이트"""
        self.tray_states[slot_id]["status"] = status
        self.tray_states[slot_id]["order_id"] = order_id

    def get_empty_slot(self):
        """빈 슬롯 찾기"""
        for slot_id, state in self.tray_states.items():
            if state["status"] == "empty":
                return slot_id
        return None

    def get_ready_slot(self):
        """서빙 준비된 슬롯 찾기"""
        for slot_id, state in self.tray_states.items():
            if state["status"] == "ready":
                return slot_id
        return None


# ========================================
# [🎛️ 좌표 튜닝 가이드]
# ========================================
"""
실제 로봇으로 다음 좌표들을 측정하여 수정하세요:

1. TRAY_LENGTH, TRAY_WIDTH:
   - 실제 트레이 크기 측정 (mm)
   
2. WORK_TABLE_POS:
   - 로봇 티칭 모드로 작업 테이블 중앙 이동
   - get_current_posx() 값 기록
   - [X, Y, Z] 입력
   
3. TRAY_SPACING_Y:
   - 트레이 2개를 나란히 놓을 간격 (mm)
   - 보통 트레이 폭 + 50mm
   
4. SERVE_TABLE_POS:
   - 손님이 가져가는 테이블 위치
   - 티칭으로 측정 후 입력

[측정 방법]
1. 로봇 티칭 모드 진입
2. 원하는 위치로 수동 이동
3. Python에서:
   from DSR_ROBOT2 import get_current_posx
   pos = get_current_posx()
   print(pos)  # [x, y, z, rx, ry, rz] 출력
4. 위 값을 코드에 입력
"""