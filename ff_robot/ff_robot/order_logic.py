import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class SlotManager:
    """
    로봇 서빙 시스템의 두뇌 역할을 하는 클래스
    - 재고 관리 (Inventory)
    - 주문 대기열 및 우선순위 관리 (FIFO)
    - 트레이 배치 좌표 계산 (Coordinate Calculation)
    - 대용량 주문 분할 처리 (Batching)
    """

    def __init__(self):
        # [수정] YOLO 학습 이름과 동일하게 변경해야 함
        self.inventory = {
            "burger1": 100,  # burger_cheese -> burger1
            "burger2": 100,
            "burger3": 100,
            "fries": 100,    # french_fries -> fries (학습명 기준)
            "nugget": 100,
            "coke": 100,
            "cider": 100
        }

        # 2. 트레이 설정
        self.MAX_CAPACITY = 4  # 트레이 하나에 담을 수 있는 최대 개수
        
        # 트레이 내부 4칸의 상대 좌표 (Offset) [x, y, z] (단위: mm)
        # Vision 마커(중심) 기준 상대 위치
        self.TRAY_OFFSETS = [
            [-50, 50, 0],   # 1번 칸 (좌상)
            [50, 50, 0],    # 2번 칸 (우상)
            [-50, -50, 0],  # 3번 칸 (좌하)
            [50, -50, 0]    # 4번 칸 (우하)
        ]

        # 3. 슬롯 상태 (0번 슬롯, 1번 슬롯)
        # 구조: { slot_id: { order_info... } }
        self.active_slots = {0: None, 1: None}
        
        # 슬롯이 꽉 찼을 때 대기하는 주문 큐
        self.pending_queue = []

    # ==========================================================================
    # [Section 1] 재고 관리 (Inventory)
    # ==========================================================================
    def check_and_deduct_stock(self, item_names, item_quantities):
        """
        주문 가능 여부 확인 및 재고 차감 (Transaction)
        Returns: (Success: bool, Message: str)
        """
        # 검증
        for name, qty in zip(item_names, item_quantities):
            if name not in self.inventory:
                return False, f"메뉴 오류: {name}"
            if self.inventory[name] < qty:
                return False, f"재고 부족: {name}"
        
        # 차감 (검증 완료 후 실행)
        for name, qty in zip(item_names, item_quantities):
            self.inventory[name] -= qty
            
        return True, "주문 접수 완료"

    def get_inventory_status(self):
        """ROS 메시지 전송용 리스트 반환"""
        return list(self.inventory.keys()), list(self.inventory.values())

    # ==========================================================================
    # [Section 2] 주문 등록 및 우선순위 관리 (Order Management)
    # ==========================================================================
    def add_order_to_slot(self, order_id, item_names, item_quantities):
        """
        확정된 주문을 슬롯 시스템에 등록
        - 빈 슬롯이 있으면 즉시 할당
        - 없으면 대기열(Queue)에 추가
        """
        # ["burger", "coke"], [2, 1] -> ["burger", "burger", "coke"] 로 변환
        full_needed_list = []
        for name, qty in zip(item_names, item_quantities):
            full_needed_list.extend([name] * qty)

        new_order = {
            "order_id": order_id,
            "needed": full_needed_list,      # 전체 필요한 리스트 (예: 10개)
            "placed_total": [],              # 지금까지 처리한 전체 리스트 (누적)
            "placed_on_tray": [],            # [중요] 현재 트레이에 놓인 리스트 (최대 4개)
            "arrival_time": time.time()      # [핵심] 선입선출(FIFO)을 위한 시간 기록
        }

        empty_slot = self._find_empty_slot()
        
        if empty_slot is not None:
            self.active_slots[empty_slot] = new_order
            print(f"[SlotManager] ✅ 주문 {order_id} -> 슬롯 {empty_slot} 배정 (우선순위: {new_order['arrival_time']})")
        else:
            self.pending_queue.append(new_order)
            print(f"[SlotManager] 💤 슬롯 만석. 주문 {order_id} 대기열 등록 (대기: {len(self.pending_queue)})")

    def get_all_needed_items(self):
        """Vision에게 요청할 타겟 리스트 추출 (현재 필요한 아이템만)"""
        needed_set = set()
        for data in self.active_slots.values():
            if data:
                # (필요한 총량) > (처리된 총량) 인 아이템만 찾음
                for item in data['needed']:
                    if data['placed_total'].count(item) < data['needed'].count(item):
                        needed_set.add(item)
        return list(needed_set)

    # ==========================================================================
    # [Section 3] 배치 로직 및 좌표 계산 (The Brain)
    # ==========================================================================
    def process_item_placement(self, detected_item, tray_pose_0, tray_pose_1):
        """
        Vision이 찾은 물건을 분석하여 배치할 위치와 액션을 결정함
        
        Args:
            detected_item (str): 감지된 물체 이름 (예: "burger")
            tray_pose_0 (list): 0번 트레이 마커 [x, y, z, rx, ry, rz]
            tray_pose_1 (list): 1번 트레이 마커
            
        Returns:
            final_pos (list): 로봇 이동 절대 좌표 [x, y, z]
            action (str): "PLACE", "PLACE_AND_SWAP", "PLACE_AND_FINISH"
        """
        # 1. 활성 주문 추출 및 [시간순 정렬] (FIFO)
        # 딕셔너리의 값들을 리스트로 변환하고 arrival_time 기준으로 정렬
        active_orders = [(sid, data) for sid, data in self.active_slots.items() if data]
        if not active_orders:
            return None, None
            
        sorted_orders = sorted(active_orders, key=lambda x: x[1]['arrival_time'])
        
        target_slot_id = None
        target_data = None

        # 2. 가장 오래된 주문부터 검사: "이 물건 필요하세요?"
        for slot_id, data in sorted_orders:
            needed = data['needed']
            placed_total = data['placed_total']
            
            # 아직 더 필요한가? (전체 필요량 > 전체 처리량)
            if needed.count(detected_item) > placed_total.count(detected_item):
                target_slot_id = slot_id
                target_data = data
                break 
        
        if target_slot_id is None:
            print(f"[SlotManager] 🤷‍♂️ '{detected_item}'는 현재 필요한 주문이 없습니다.")
            return None, None

        # 3. 해당 슬롯의 트레이 좌표 확보
        current_tray_pose = tray_pose_0 if target_slot_id == 0 else tray_pose_1
        
        if current_tray_pose is None:
            print(f"[SlotManager] 🚨 오류: 슬롯 {target_slot_id}의 트레이 마커를 못 찾았습니다.")
            return None, "ERROR_NO_TRAY"

        # 4. 좌표 계산 (현재 트레이에 몇 개 놓였는지 확인하여 오프셋 적용)
        # placed_on_tray의 길이를 보면 이번에 몇 번째 칸(0~3)에 놓을지 알 수 있음
        current_idx = len(target_data['placed_on_tray'])
        final_pos = self._calculate_absolute_pose(current_tray_pose, current_idx)
        
        if final_pos is None:
             return None, "ERROR_CALC"

        # 5. 데이터 업데이트 (장부 기록)
        target_data['placed_total'].append(detected_item)
        target_data['placed_on_tray'].append(detected_item)
        
        # 6. 액션(상태) 결정 - 대용량 주문 분할 로직 포함
        action = "PLACE"
        
        is_order_done = len(target_data['placed_total']) == len(target_data['needed'])
        is_tray_full = len(target_data['placed_on_tray']) >= self.MAX_CAPACITY

        if is_order_done:
            # 주문이 완전히 끝났을 때
            action = "PLACE_AND_FINISH"
            print(f"[SlotManager] 🎉 주문 {target_data['order_id']} 완료! (총 {len(target_data['placed_total'])}개)")
            # (주의: 여기서 바로 clear_slot을 호출하지 않고, 로봇 서빙 동작 완료 후 호출)
            
        elif is_tray_full:
            # 주문은 남았는데 트레이가 꽉 찼을 때 (분할 서빙)
            action = "PLACE_AND_SWAP"
            print(f"[SlotManager] 🔄 주문 {target_data['order_id']} 트레이 교체 (4개 꽉 참, 남은 수량 처리 대기)")
            
            # [핵심 로직] 논리적으로 트레이를 비워줍니다.
            # 그래야 다음 아이템이 올 때 다시 0번 칸(첫 번째 칸)부터 좌표가 계산됩니다.
            target_data['placed_on_tray'] = [] 

        return final_pos, action

    def clear_slot(self, slot_id):
        """
        서빙이 끝난 슬롯을 비우고, 대기 중인 주문이 있다면 가져옴
        (로봇 노드에서 PLACE_AND_FINISH 액션 수행 후 호출해야 함)
        """
        print(f"[SlotManager] 🧹 슬롯 {slot_id} 정리 완료.")
        self.active_slots[slot_id] = None
        
        if self.pending_queue:
            next_order = self.pending_queue.pop(0)
            print(f"[SlotManager] 📥 대기열의 주문 {next_order['order_id']}를 슬롯 {slot_id}로 가져옵니다.")
            # 새 슬롯에 할당
            self.active_slots[slot_id] = next_order

    # ==========================================================================
    # [Section 4] 내부 헬퍼 함수 (Private Helpers)
    # ==========================================================================
    def _find_empty_slot(self):
        for sid, data in self.active_slots.items():
            if data is None: return sid
        return None

    def _calculate_absolute_pose(self, tray_pose, offset_idx):
        """
        행렬 연산: 트레이 절대 좌표 * 오프셋 = 최종 로봇 좌표
        tray_pose: [x, y, z, rx, ry, rz] (단위: mm, degree)
        """
        try:
            tx, ty, tz = tray_pose[:3]
            rx, ry, rz = tray_pose[3:]

            # 1. 회전 행렬 생성 (Euler -> Rotation Matrix)
            r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
            rot_mat = r.as_matrix()

            # 2. 변환 행렬 (Homogeneous Matrix 4x4) 구성
            T_mat = np.eye(4)
            T_mat[:3, :3] = rot_mat
            T_mat[:3, 3] = [tx, ty, tz]

            # 3. 오프셋 벡터 (4x1)
            ox, oy, oz = self.TRAY_OFFSETS[offset_idx]
            offset_vec = np.array([ox, oy, oz, 1])

            # 4. 행렬 곱셈 (Transform)
            final_vec = T_mat @ offset_vec
            
            return list(final_vec[:3]) # [x, y, z] 반환
            
        except Exception as e:
            print(f"[SlotManager] 좌표 계산 오류: {e}")
            return None
