# order_logic.py

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class SlotManager:
    def __init__(self):
        # 1. 재고 현황
        self.inventory = {
            "burger1": 100, "burger2": 100, "burger3": 100,
            "fries": 100, "nugget": 100,
            "coke": 100, "cider": 100
        }

        self.MAX_CAPACITY = 4
        self.active_slots = {0: None, 1: None}
        self.pending_queue = []

        # ---------------------------------------------------------
        # [NEW] 트레이 배치 패턴 정의 (16cm x 22cm 기준)
        # ---------------------------------------------------------
        # DX = 40mm (가로 4등분 지점), DY = 55mm (세로 4등분 지점)
        DX = 40.0
        DY = 55.0
        
        self.LAYOUT_PATTERNS = {
            # 1개: 정중앙 (0, 0)
            1: [
                [0.0, 0.0, 0.0]
            ],
            
            # 2개: 위(+y), 아래(-y)로 2분할 (중앙 정렬)
            2: [
                [0.0,  DY, 0.0],  # 위쪽 (Top)
                [0.0, -DY, 0.0]   # 아래쪽 (Bottom)
            ],
            
            # 3개: 위쪽은 1개(중앙), 아래쪽은 2개(좌우)로 분할
            3: [
                [0.0,  DY, 0.0],  # 위쪽 중앙
                [-DX, -DY, 0.0],  # 아래쪽 왼쪽 (-x, -y)
                [ DX, -DY, 0.0]   # 아래쪽 오른쪽 (+x, -y)
            ],
            
            # 4개: 4등분 (좌상, 우상, 좌하, 우하)
            4: [
                [-DX,  DY, 0.0],  # 좌상 (-x, +y)
                [ DX,  DY, 0.0],  # 우상 (+x, +y)
                [-DX, -DY, 0.0],  # 좌하 (-x, -y)
                [ DX, -DY, 0.0]   # 우하 (+x, -y)
            ]
        }

    # ==========================================================================
    # [Section 1] 재고 관리
    # ==========================================================================
    def check_and_deduct_stock(self, item_names, item_quantities):
        for name, qty in zip(item_names, item_quantities):
            if name not in self.inventory: return False, f"메뉴 오류: {name}"
            if self.inventory[name] < qty: return False, f"재고 부족: {name}"
        
        for name, qty in zip(item_names, item_quantities):
            self.inventory[name] -= qty
        return True, "주문 접수 완료"

    # ==========================================================================
    # [Section 2] 주문 등록 및 관리
    # ==========================================================================
    def add_order_to_slot(self, order_id, item_names, item_quantities):
        full_needed_list = []
        for name, qty in zip(item_names, item_quantities):
            full_needed_list.extend([name] * qty)

        new_order = {
            "order_id": order_id,
            "needed": full_needed_list,
            "placed_total": [],
            "arrival_time": time.time()
        }

        empty_slot = self._find_empty_slot()
        if empty_slot is not None:
            self.active_slots[empty_slot] = new_order
            print(f"[SlotManager] ✅ 주문 {order_id} -> 슬롯 {empty_slot} 배정")
        else:
            self.pending_queue.append(new_order)
            print(f"[SlotManager] 💤 슬롯 만석. 대기열 등록 (대기: {len(self.pending_queue)})")

    def get_all_needed_items(self):
        """우선순위(도착 시간) 순서대로 필요한 아이템 리스트 반환"""
        final_list = []
        active_orders = [data for data in self.active_slots.values() if data]
        sorted_orders = sorted(active_orders, key=lambda x: x['arrival_time'])
        
        for data in sorted_orders:
            needed_types = set(data['needed'])
            for item in needed_types:
                if data['placed_total'].count(item) < data['needed'].count(item):
                    final_list.append(item)
        return final_list

    def peek_target_slot(self, item_name):
        """아이템을 차감하지 않고, 어느 슬롯에 할당될 예정인지 반환"""
        active_orders = [(sid, data) for sid, data in self.active_slots.items() if data]
        sorted_orders = sorted(active_orders, key=lambda x: x[1]['arrival_time'])

        for sid, data in sorted_orders:
            needed_cnt = data['needed'].count(item_name)
            placed_cnt = data['placed_total'].count(item_name)
            if placed_cnt < needed_cnt:
                return sid 
        return 0

    # ==========================================================================
    # [Section 3] 배치 좌표 계산 (New)
    # ==========================================================================
    def get_tray_place_pose(self, tray_center_pose, current_idx, total_count):
        """
        총 개수(total_count)에 따라 미리 정의된 패턴 좌표를 반환
        """
        # 예외 처리: 4개 초과 시 4개 패턴 반복 사용
        if total_count > 4: total_count = 4 
        
        # 패턴 가져오기
        pattern = self.LAYOUT_PATTERNS.get(total_count, self.LAYOUT_PATTERNS[4])
        
        # 인덱스 안전장치
        safe_idx = current_idx % len(pattern)
        off_x, off_y, off_z = pattern[safe_idx]

        # 트레이 중심에 오프셋 적용
        tx, ty, tz, trx, try_, trz = tray_center_pose
        
        final_x = tx + off_x
        final_y = ty + off_y
        final_z = tz + off_z
        
        return [final_x, final_y, final_z, trx, try_, trz]

    # ==========================================================================
    # [Section 4] 상태 업데이트
    # ==========================================================================
    def process_item_and_check_complete(self, item_name):
        """아이템 처리 후 완료 여부 반환"""
        active_orders = [(sid, data) for sid, data in self.active_slots.items() if data]
        sorted_orders = sorted(active_orders, key=lambda x: x[1]['arrival_time'])

        target_slot_id = None
        target_data = None

        for sid, data in sorted_orders:
            needed_cnt = data['needed'].count(item_name)
            placed_cnt = data['placed_total'].count(item_name)
            if placed_cnt < needed_cnt:
                target_slot_id = sid
                target_data = data
                break
        
        if target_slot_id is None:
            print(f"[SlotManager] ⚠️ '{item_name}'는 현재 필요한 주문이 없습니다.")
            return None, False

        target_data['placed_total'].append(item_name)
        print(f"[SlotManager] 📦 Slot {target_slot_id}: '{item_name}' 처리됨. ({len(target_data['placed_total'])}/{len(target_data['needed'])})")

        if len(target_data['placed_total']) >= len(target_data['needed']):
            return target_slot_id, True
        else:
            return target_slot_id, False

    def clear_slot(self, slot_id):
        print(f"[SlotManager] 🧹 슬롯 {slot_id} 주문 종료 및 초기화.")
        self.active_slots[slot_id] = None
        if self.pending_queue:
            next_order = self.pending_queue.pop(0)
            self.active_slots[slot_id] = next_order
            print(f"[SlotManager] 📥 대기열 주문 {next_order['order_id']} -> 슬롯 {slot_id} 할당")

    def _find_empty_slot(self):
        for sid, data in self.active_slots.items():
            if data is None: return sid
        return None
    
    # (참고) 추후 트레이 좌표 계산을 위해 남겨둠 (지금은 사용 안함)
    def calculate_tray_coordinate(self, tray_pose, index):
        # 여기에 좌표 변환 로직 구현 가능
        pass
