import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class SlotManager:
    def __init__(self):
        # --- 설정값 (Config) ---
        self.MAX_CAPACITY = 4  # 트레이 한 판에 담을 수 있는 최대 개수
        
        # 트레이 내부 4칸의 상대 좌표 (Offset) [x, y, z] (단위: mm 가정)
        # 트레이 중심(또는 마커) 기준
        self.TRAY_OFFSETS = [
            [-50, 50, 0],   # 1번 칸 (좌상)
            [50, 50, 0],    # 2번 칸 (우상)
            [-50, -50, 0],  # 3번 칸 (좌하)
            [50, -50, 0]    # 4번 칸 (우하)
        ]

        # --- 상태 변수 (State) ---
        # 0번 슬롯, 1번 슬롯 (2개의 트레이 공간)
        self.active_slots = {0: None, 1: None}
        
        # 슬롯이 꽉 찼을 때 대기하는 주문 큐
        self.pending_queue = []

    def add_order(self, order_id, item_list):
        """
        주문이 들어오면 빈 슬롯에 배정하거나 대기열에 추가함
        """
        # 주문 정보 구조체 생성
        new_order = {
            "order_id": order_id,
            "needed": item_list,         # 전체 필요한 아이템 리스트
            "placed_total": [],          # 지금까지 놓은 전체 아이템
            "placed_on_tray": [],        # 현재 트레이에 놓인 아이템 (최대 4개)
            "arrival_time": time.time()  # [중요] 우선순위용 시간 기록
        }

        empty_slot = self._find_empty_slot()
        
        if empty_slot is not None:
            self.active_slots[empty_slot] = new_order
            print(f"[Manager] ✅ 주문 {order_id} -> 슬롯 {empty_slot} 배정 완료")
        else:
            self.pending_queue.append(new_order)
            print(f"[Manager] 💤 슬롯 꽉 참. 주문 {order_id} 대기열 등록 (대기: {len(self.pending_queue)})")

    def get_all_needed_items(self):
        """Vision에게 요청할 타겟 리스트 추출 (중복 제거)"""
        needed_set = set()
        for data in self.active_slots.values():
            if data:
                # 필요한 것 - 이미 전체 놓은 것 (차집합 개념)
                # 실제로는 리스트 count 비교가 정확하지만, 단순화를 위해 set으로 표현
                # Vision에게는 "종류"만 알려주면 되므로 set이 효율적
                for item in data['needed']:
                    if data['placed_total'].count(item) < data['needed'].count(item):
                        needed_set.add(item)
        return list(needed_set)

    def process_item_placement(self, detected_item, tray_pose_0, tray_pose_1):
        """
        [핵심 함수] Vision이 찾은 물건을 어디에 놓을지 결정하고, 절대 좌표를 계산해서 반환
        
        Args:
            detected_item (str): 찾은 물건 이름 (예: "burger")
            tray_pose_0 (list): 0번 트레이 마커 좌표 [x, y, z, rx, ry, rz]
            tray_pose_1 (list): 1번 트레이 마커 좌표
            
        Returns:
            final_pos (list): 로봇이 이동해야 할 절대 좌표 [x, y, z]
            action (str): 수행해야 할 후속 동작 ("PLACE", "SWAP", "FINISH")
        """
        
        # 1. 우선순위 정렬 (먼저 온 주문 순서)
        active_orders = [(sid, data) for sid, data in self.active_slots.items() if data]
        sorted_orders = sorted(active_orders, key=lambda x: x[1]['arrival_time'])
        
        target_slot_id = None
        target_data = None

        # 2. 누구 건지 찾기
        for slot_id, data in sorted_orders:
            needed = data['needed']
            placed_total = data['placed_total']
            
            # 이 주문에 아직 이 물건이 필요한가?
            if needed.count(detected_item) > placed_total.count(detected_item):
                target_slot_id = slot_id
                target_data = data
                break
        
        if target_slot_id is None:
            print(f"[Manager] 🤷‍♂️ '{detected_item}'는 지금 필요한 주문이 없습니다.")
            return None, None

        # 3. 해당 슬롯의 현재 트레이 좌표 가져오기
        current_tray_pose = tray_pose_0 if target_slot_id == 0 else tray_pose_1
        
        if current_tray_pose is None:
            print(f"[Manager] 🚨 오류: 슬롯 {target_slot_id}의 트레이 마커를 못 찾았습니다!")
            return None, "ERROR_NO_TRAY"

        # 4. 좌표 계산 (트레이 내 몇 번째 칸인가?)
        current_idx = len(target_data['placed_on_tray'])
        final_pos = self._calculate_absolute_pose(current_tray_pose, current_idx)
        
        # 5. 데이터 업데이트
        target_data['placed_total'].append(detected_item)
        target_data['placed_on_tray'].append(detected_item)
        
        # 6. 액션 결정 (트레이 관리)
        action = "PLACE"
        
        # 트레이가 꽉 찼거나, 주문이 다 끝났거나
        is_tray_full = len(target_data['placed_on_tray']) >= self.MAX_CAPACITY
        is_order_done = len(target_data['placed_total']) == len(target_data['needed'])
        
        if is_order_done:
            action = "PLACE_AND_FINISH"
            print(f"[Manager] 🎉 주문 {target_data['order_id']} 완료! 서빙합니다.")
        elif is_tray_full:
            action = "PLACE_AND_SWAP"
            target_data['placed_on_tray'] = [] # 트레이 비우기 (논리적)
            print(f"[Manager] 🔄 주문 {target_data['order_id']} 트레이 교체 필요.")

        return final_pos, action

    def clear_slot(self, slot_id):
        """서빙 완료 후 슬롯 초기화 및 대기열 당겨오기"""
        print(f"[Manager] 🧹 슬롯 {slot_id} 청소 완료.")
        self.active_slots[slot_id] = None
        
        if self.pending_queue:
            next_order = self.pending_queue.pop(0)
            # 재귀적으로 다시 배정 (빈 슬롯이 생겼으므로 바로 들어감)
            self.add_order(next_order['order_id'], next_order['needed'])

    def _find_empty_slot(self):
        for sid, data in self.active_slots.items():
            if data is None: return sid
        return None

    def _calculate_absolute_pose(self, tray_pose, offset_idx):
        """행렬 연산을 통한 상대 좌표 -> 절대 좌표 변환"""
        # tray_pose: [x, y, z, rx, ry, rz] (단위: mm, degree)
        tx, ty, tz = tray_pose[:3]
        rx, ry, rz = tray_pose[3:]

        # 1. 회전 행렬 생성
        r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        rot_mat = r.as_matrix()

        # 2. 변환 행렬 (Homogeneous Matrix 4x4) 구성
        T_mat = np.eye(4)
        T_mat[:3, :3] = rot_mat
        T_mat[:3, 3] = [tx, ty, tz]

        # 3. 오프셋 벡터 (4x1)
        ox, oy, oz = self.TRAY_OFFSETS[offset_idx]
        offset_vec = np.array([ox, oy, oz, 1])

        # 4. 행렬 곱셈
        final_vec = T_mat @ offset_vec
        
        return list(final_vec[:3]) # [x, y, z] 반환