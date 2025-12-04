# order_logic.py
# [Final Complete Version]
# - 배치 패턴(LAYOUT_PATTERNS) 적용
# - mark_item_done 함수 포함
# - 불필요한 행렬 연산 제거로 최적화됨

import time
import numpy as np

class SlotManager:
    """
    로봇 서빙 시스템의 상태 관리자
    - 재고 관리, 주문 대기열(FIFO), 슬롯 상태 추적
    - 트레이 내 배치 좌표 계산 (Dynamic Layout)
    """

    def __init__(self):
        # 1. 재고 현황
        self.inventory = {
            "burger1": 100, "burger2": 100, "burger3": 100,
            "fries": 100, "nugget": 100,
            "coke": 100, "cider": 100
        }

        self.MAX_CAPACITY = 4
        
        # 0번, 1번 슬롯 초기화
        # 구조: { slot_id: { order_id, needed:[], placed_total:[], ... } }
        self.active_slots = {0: None, 1: None}
        self.pending_queue = []

        # ---------------------------------------------------------
        # [배치 패턴 정의] 트레이 내 물건 배치 오프셋 (단위: mm)
        # ---------------------------------------------------------
        # 트레이 크기(16x22cm) 고려: 가로 40mm, 세로 55mm 간격
        DX = 40.0
        DY = 55.0
        
        # 아이템 총 개수(Key)에 따른 좌표 리스트(Value)
        self.LAYOUT_PATTERNS = {
            # 1개: 정중앙
            1: [
                [0.0, 0.0, 0.0]
            ],
            
            # 2개: 위/아래로 나란히
            2: [
                [0.0,  DY, 0.0],  # 위
                [0.0, -DY, 0.0]   # 아래
            ],
            
            # 3개: 위 1개, 아래 2개
            3: [
                [0.0,  DY, 0.0],  # 위 중앙
                [-DX, -DY, 0.0],  # 아래 왼쪽
                [ DX, -DY, 0.0]   # 아래 오른쪽
            ],
            
            # 4개: 4등분 격자
            4: [
                [-DX,  DY, 0.0],  # 좌상
                [ DX,  DY, 0.0],  # 우상
                [-DX, -DY, 0.0],  # 좌하
                [ DX, -DY, 0.0]   # 우하
            ]
        }

    # ==========================================================================
    # [Section 1] 재고 및 주문 접수
    # ==========================================================================
    def check_and_deduct_stock(self, item_names, item_quantities):
        """재고 확인 및 차감"""
        for name, qty in zip(item_names, item_quantities):
            if name not in self.inventory: return False, f"메뉴 오류: {name}"
            if self.inventory[name] < qty: return False, f"재고 부족: {name}"
        
        for name, qty in zip(item_names, item_quantities):
            self.inventory[name] -= qty
        return True, "주문 접수 완료"

    def add_order_to_slot(self, order_id, item_names, item_quantities):
        """주문을 슬롯에 할당하거나 대기열에 추가"""
        # 입력된 리스트를 하나로 펼침 (예: burger 2개 -> [burger, burger])
        full_needed_list = []
        for name, qty in zip(item_names, item_quantities):
            full_needed_list.extend([name] * qty)

        new_order = {
            "order_id": order_id,
            "needed": full_needed_list,      # 필요한 전체 목록
            "placed_total": [],              # 처리 완료된 목록
            "arrival_time": time.time()      # FIFO를 위한 타임스탬프
        }

        empty_slot = self._find_empty_slot()
        if empty_slot is not None:
            self.active_slots[empty_slot] = new_order
            print(f"[SlotManager] ✅ 주문 {order_id} -> 슬롯 {empty_slot} 배정")
        else:
            self.pending_queue.append(new_order)
            print(f"[SlotManager] 💤 슬롯 만석. 대기열 등록 (대기: {len(self.pending_queue)})")

    # ==========================================================================
    # [Section 2] 작업 조회 (Robot Controller가 사용)
    # ==========================================================================
    def get_all_needed_items(self):
        """
        우선순위(도착 시간) 순서대로 현재 필요한 아이템 리스트 반환
        """
        final_list = []
        # 활성 주문만 추출
        active_orders = [data for data in self.active_slots.values() if data]
        # 도착 시간 순 정렬 (먼저 온 주문이 앞쪽으로)
        sorted_orders = sorted(active_orders, key=lambda x: x['arrival_time'])
        
        for data in sorted_orders:
            # 중복 제거하여 필요한 종류 확인
            needed_types = set(data['needed'])
            for item in needed_types:
                # 아직 덜 채운 아이템만 추가
                if data['placed_total'].count(item) < data['needed'].count(item):
                    final_list.append(item)
                    
        return final_list
    

    def get_tray_place_pose(self, tray_center_pose, current_idx, total_count):
        """
        트레이 중심 좌표와 순서를 받아, 실제 놓을 좌표(Offset 적용)를 반환
        """
        # 예외 처리: 4개 초과 시 4개 패턴 반복
        if total_count > 4: total_count = 4 
        
        # 패턴 가져오기
        pattern = self.LAYOUT_PATTERNS.get(total_count, self.LAYOUT_PATTERNS[4])
        
        # 인덱스 안전장치
        safe_idx = current_idx % len(pattern)
        off_x, off_y, off_z = pattern[safe_idx]

        # 트레이 중심에 오프셋 더하기
        tx, ty, tz, trx, try_, trz = tray_center_pose
        
        return [tx + off_x, ty + off_y, tz + off_z, trx, try_, trz]

    # ==========================================================================
    # [Section 3] 완료 처리 및 상태 관리
    # ==========================================================================
    def mark_item_done(self, slot_id, item_name):
        """
        [핵심] 아이템 처리를 확정(저장)하고, 주문이 끝났는지(True/False) 반환
        """
        if slot_id not in self.active_slots or self.active_slots[slot_id] is None:
            return False

        data = self.active_slots[slot_id]
        
        # 1. 처리 내역 기록
        data['placed_total'].append(item_name)
        print(f"[SlotManager] ✅ Slot {slot_id}: '{item_name}' 처리됨. ({len(data['placed_total'])}/{len(data['needed'])})")

        # 2. 주문 완료 여부 검사
        if len(data['placed_total']) >= len(data['needed']):
            print(f"[SlotManager] 🎉 Slot {slot_id} 주문 완성!")
            return True # 완료됨 -> 서빙 시작 신호
        
        return False # 아직 남음
    
    def peek_target_slot(self, item_name):
        """
        아이템을 실제로 차감하지 않고, 이 아이템이 어느 슬롯으로 갈지 미리 확인
        (트레이 위치 결정을 위해 사용)
        """
        active_orders = [(sid, data) for sid, data in self.active_slots.items() if data]
        # 도착 시간 순 정렬
        sorted_orders = sorted(active_orders, key=lambda x: x[1]['arrival_time'])

        for sid, data in sorted_orders:
            # 이 주문에 해당 아이템이 더 필요한가?
            if data['placed_total'].count(item_name) < data['needed'].count(item_name):
                return sid 
        return 0 # 기본값 (예외 상황)

    def clear_slot(self, slot_id):
        """슬롯을 비우고 대기열에 있는 주문을 당겨옴"""
        print(f"[SlotManager] 🧹 Slot {slot_id} Cleared (서빙 완료).")
        self.active_slots[slot_id] = None
        
        if self.pending_queue:
            next_order = self.pending_queue.pop(0)
            self.active_slots[slot_id] = next_order
            print(f"[SlotManager] 📥 대기열 주문 {next_order['order_id']} -> Slot {slot_id} 할당")

    def _find_empty_slot(self):
        for sid, data in self.active_slots.items():
            if data is None: return sid
        return None
