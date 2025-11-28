import time  # [필수] 시간 지연을 위해 추가

class GripperManager:
    def __init__(self, gripper):
        self.gripper = gripper

        # 메뉴별 그리퍼 파라미터 (단위: 1/10mm, N)
        self.params = {
            "burger1": {"open": 650 + 100, "force": 40},
            "burger2": {"open": 650 + 100, "force": 40},
            "burger3": {"open": 650 + 100, "force": 40},
            "fries":   {"open": 900 + 100, "force": 30}, 
            "nugget":  {"open": 900 + 100, "force": 30},
            "coke":    {"open": 675 + 100, "force": 60}, 
            "cider":   {"open": 675 + 100, "force": 60},
        }

    def get_params(self, item):
        """메뉴 파라미터 가져오기 (없으면 동작 중단)"""
        p = self.params.get(item)
        if p is None:
            print(f"⚠️ [Gripper] '{item}' 파라미터 없음. 기본값 사용.")
            return {"open": 800, "force": 40}
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

        # [추가] 실제로 벌어질 때까지 대기
        time.sleep(0.5)

    def execute_grip(self, item):
        """지정된 힘으로 물기"""
        p = self.get_params(item)
        force = p["force"]

        try:
            # force 파라미터 지원 여부 확인 후 호출
            try:
                self.gripper.close_gripper(force)
                print(f"✊ [{item}] 집기 완료 (힘: {force}N)")
            except TypeError:
                self.gripper.close_gripper()
                print(f"✊ [{item}] 집기 완료 (기본 힘)")
        except Exception as e:
            print(f"   ❌ 그리퍼 클로즈 실패: {e}")

        # [추가] 꽉 잡을 때까지 충분히 대기 (중요: 놓치지 않으려면 길게)
        time.sleep(1.5)

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

        # [추가] 물건이 떨어질 때까지 대기
        time.sleep(0.5)