import time

class GripperManager:
    def __init__(self, gripper):
        self.gripper = gripper

        # [수정] 메뉴별 그리퍼 파라미터 (단위: 1/10mm)
        # 'force' 대신 'close' (닫을 때 목표 너비)를 사용합니다.
        # 예: 600 = 60mm
        self.params = {
            # 버거 (예: 실제 크기 70mm -> 잡는 너비 55mm로 설정하여 꽉 잡음)
            "burger1": {"open": 800, "close": 600}, 
            "burger2": {"open": 800, "close": 600},
            "burger3": {"open": 800, "close": 600},
            
            # 사이드 (감자튀김, 너겟 등)
            "fries":   {"open": 1200, "close": 850}, 
            "nugget":  {"open": 1200, "close": 850},
            
            # 음료 (캔 지름 약 66mm -> 62~63mm로 설정하여 적당히 텐션 유지)
            "coke":    {"open": 850, "close": 640}, 
            "cider":   {"open": 850, "close": 640},
        }

    def get_params(self, item):
        """메뉴 파라미터 가져오기"""
        p = self.params.get(item)
        if p is None:
            print(f"⚠️ [Gripper] '{item}' 파라미터 없음. 기본값 사용.")
            return {"open": 800, "close": 500}
        return p

    def prepare_grip(self, item):
        """집기 전에 그리퍼 오픈"""
        p = self.get_params(item)
        open_width = p["open"]
        
        try:
            if hasattr(self.gripper, 'move_gripper'):
                self.gripper.move_gripper(open_width)
                print(f"👐 [{item}] 그리퍼 준비: 너비 {open_width}")
            else:
                self.gripper.open_gripper()
                print("👐 [{item}] 그리퍼 준비: 최대 오픈")
        except Exception as e:
            print(f"   ❌ 그리퍼 오픈 실패: {e}")

        # 이동 시간 대기 (0.5초면 충분)
        time.sleep(0.5)

    def execute_grip(self, item):
        """
        [핵심 수정] 힘(Force) 대신 위치(Position)로 잡기
        지정된 'close' 너비로 빠르게 이동합니다.
        """
        p = self.get_params(item)
        close_width = p["close"]

        try:
            if hasattr(self.gripper, 'move_gripper'):
                # 지정된 너비로 이동 (물체가 있으면 그 크기에서 멈춤)
                self.gripper.move_gripper(close_width)
                print(f"✊ [{item}] 그리퍼 닫기: 목표 너비 {close_width}")
            else:
                self.gripper.close_gripper()
                print(f"✊ [{item}] 그리퍼 닫기: 기본 동작")
        except Exception as e:
            print(f"   ❌ 그리퍼 클로즈 실패: {e}")

        # [대기 시간 수정] 
        # move_gripper는 빠르므로 2.0초까지 기다릴 필요 없음. 1.0초면 충분.
        time.sleep(1.0)

    def release(self, item):
        """놓기"""
        p = self.get_params(item)
        open_width = p["open"]
        
        try:
            if hasattr(self.gripper, 'move_gripper'):
                self.gripper.move_gripper(open_width)
            else:
                self.gripper.open_gripper()
            print("👐 오브젝트 해제")
        except Exception as e:
            print(f"   ❌ 릴리즈 실패: {e}")

        time.sleep(0.5)