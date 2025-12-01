# order_logic.py

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class SlotManager:
    """
    로봇 서빙 시스템의 두뇌 역할을 하는 클래스
    - 재고 관리 (Inventory)
    - 주문 대기열 및 우선순위 관리 (FIFO)
    - 주문 상태 추적 (State Management)
    """

    def __init__(self):
        # 1. 재고 현황
        self.inventory = {
            "burger1": 100, "burger2": 100, "burger3": 100,
            "fries": 100, "nugget": 100,
            "coke": 100, "cider": 100
        }

        # 2. 트레이 설정 (추후 트레이 좌표 계산용)
        self.MAX_CAPACITY = 4
        self.TRAY_OFFSETS = [
            [-50, 50, 0], [50, 50, 0], 
            [-50, -50, 0], [50, -50, 0]
        ]

        # 3. 슬롯 상태 (0번, 1번)
        # 구조: { slot_id: { order_id, needed:[], placed_total:[], ... } }
        self.active_slots = {0: None, 1: None}
        self.pending_queue = []

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
            "needed": full_needed_list,      # 전체 필요한 리스트 (예: ['fries', 'coke'])
            "placed_total": [],              # 처리 완료된 리스트
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
        """현재 활성 슬롯에서 아직 처리되지 않은 아이템 리스트 반환"""
        needed_set = set()
        for data in self.active_slots.values():
            if data:
                # 필요한 개수 > 처리된 개수 인 항목만 추출
                for item in data['needed']:
                    if data['placed_total'].count(item) < data['needed'].count(item):
                        needed_set.add(item)
        return list(needed_set)

    # ==========================================================================
    # [Section 3] 상태 업데이트 (핵심 수정됨)
    # ==========================================================================
    def mark_item_done(self, slot_id, item_name):
        """
        아이템 하나를 처리 완료로 표시하고, 
        주문 전체가 완료되었는지 여부(True/False)를 반환
        """
        if slot_id not in self.active_slots or self.active_slots[slot_id] is None:
            return True # 이미 비어있거나 없음

        data = self.active_slots[slot_id]
        
        # 1. 처리 목록에 추가 (유효성 검사 포함)
        needed_count = data['needed'].count(item_name)
        current_count = data['placed_total'].count(item_name)

        if current_count < needed_count:
            data['placed_total'].append(item_name)
            print(f"[SlotManager] 📦 '{item_name}' 처리됨. ({len(data['placed_total'])}/{len(data['needed'])})")
        else:
            print(f"[SlotManager] ⚠️ '{item_name}'는 이미 모두 처리되었습니다.")

        # 2. 전체 주문 완료 여부 확인
        if len(data['placed_total']) >= len(data['needed']):
            return True # 주문 완료
        else:
            return False # 아직 남음

    def clear_slot(self, slot_id):
        """슬롯 비우기 및 대기열 당겨오기"""
        print(f"[SlotManager] 🧹 슬롯 {slot_id} 주문 종료 및 초기화.")
        self.active_slots[slot_id] = None
        
        if self.pending_queue:
            next_order = self.pending_queue.pop(0)
            self.active_slots[slot_id] = next_order
            print(f"[SlotManager] 📥 대기열 주문 {next_order['order_id']} -> 슬롯 {slot_id} 할당")

    # ==========================================================================
    # [Section 4] 유틸리티
    # ==========================================================================
    def _find_empty_slot(self):
        for sid, data in self.active_slots.items():
            if data is None: return sid
        return None
    
    # (참고) 추후 트레이 좌표 계산을 위해 남겨둠 (지금은 사용 안함)
    def calculate_tray_coordinate(self, tray_pose, index):
        # 여기에 좌표 변환 로직 구현 가능
        pass
