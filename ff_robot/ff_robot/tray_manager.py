#!/usr/bin/env python3
# tray_manager.py
# [수정] 잡기 오프셋 방향 반대로 변경 (로봇 가까운 쪽 잡기)

import numpy as np

class TrayManager:
    def __init__(self):
        # [CHECK] 실제 측정값 확인 필수
        self.TRAY_LENGTH = 260.0  
        self.TRAY_WIDTH = 205.0   
        self.TRAY_GRIP_OFFSET_X = self.TRAY_LENGTH / 2.0  # 102.5mm
        
        # 작업 테이블 위치
        self.WORK_TABLE_POS = [205.0, 20.0, 25.0]  
        self.WORK_TABLE_ORIENTATION = [43.35, -180.0, -134.82] 
        
        self.USE_INDIVIDUAL_SLOT_ORIENTATION = True
        self.SLOT1_ORIENTATION = [43.35, -180.0, -134.82]   
        self.SLOT2_ORIENTATION = [35.09, 179.78, 141.93]     
        
        self.TRAY_SPACING_X = 230
        self.SERVE_TABLE_OFFSET_Y = 260.0
        
        self.tray_states = {
            0: {"status": "empty", "order_id": None}, 
            1: {"status": "empty", "order_id": None}   
        }

    def calculate_tray_grip_point(self, tray_center_pos, tray_rotation_rz):
        cx, cy, cz = tray_center_pos
        
        rot_rad = np.radians(tray_rotation_rz)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        
        # [수정됨] 마이너스(-) 부호를 붙여서 로봇 쪽(가까운 쪽) 끄트머리를 계산합니다.
        # 이전: [self.TRAY_GRIP_OFFSET_X, 0.0, 0.0] -> 900mm 초과 원인
        local_offset_x = -self.TRAY_GRIP_OFFSET_X 
        local_offset_y = 0.0
        
        # 회전 적용
        global_offset_x = local_offset_x * cos_r - local_offset_y * sin_r
        global_offset_y = local_offset_x * sin_r + local_offset_y * cos_r
        
        grip_x = cx + global_offset_x
        grip_y = cy + global_offset_y
        grip_z = cz 
        
        grip_rx = 0.0
        grip_ry = 180.0
        grip_rz = tray_rotation_rz 
        
        return [grip_x, grip_y, grip_z, grip_rx, grip_ry, grip_rz]

    def calculate_work_table_position(self, slot_id, tray_rotation_rz=0.0):
        base_x, base_y, base_z = self.WORK_TABLE_POS
        offset_x = (slot_id) * self.TRAY_SPACING_X 
        
        if self.USE_INDIVIDUAL_SLOT_ORIENTATION:
            if slot_id == 0:
                base_rx, base_ry, base_rz = self.SLOT1_ORIENTATION
            else:
                base_rx, base_ry, base_rz = self.SLOT2_ORIENTATION
        else:
            base_rx, base_ry, base_rz = self.WORK_TABLE_ORIENTATION
            
        return [base_x + offset_x, base_y, base_z, base_rx, base_ry, base_rz]

    def calculate_serve_position(self, slot_id=0):
        work_pos = self.calculate_work_table_position(slot_id)
        return [
            work_pos[0],
            work_pos[1] + self.SERVE_TABLE_OFFSET_Y,
            work_pos[2],
            work_pos[3],
            work_pos[4],
            work_pos[5]
        ]

    def update_tray_status(self, slot_id, status, order_id=None):
        self.tray_states[slot_id]["status"] = status
        if order_id is not None:
            self.tray_states[slot_id]["order_id"] = order_id

    def get_empty_slot(self):
        for slot_id, state in self.tray_states.items():
            if state["status"] == "empty": return slot_id
        return None
    
    def get_ready_slot(self):
        for slot_id, state in self.tray_states.items():
            if state["status"] == "ready": return slot_id
        return None