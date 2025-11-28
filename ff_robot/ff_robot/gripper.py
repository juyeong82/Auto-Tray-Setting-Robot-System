class GripperManager:
    def __init__(self, gripper):
        self.gripper = gripper

        # 메뉴별 그리퍼 파라미터
        self.params = {
            "burger1": {"open": 650 + 100, "force": 30},
            "burger2": {"open": 650 + 100, "force": 30},
            "burger3": {"open": 650 + 100, "force": 30},
            "fries":   {"open": 900 + 100,  "force": 30},
            "nugget":  {"open": 900 + 100,  "force": 30},
            "coke":    {"open": 675 + 100,  "force": 70},
            "cider":   {"open": 675 + 100,  "force": 70},
        }

    def get_params(self, item):
        """메뉴 파라미터 가져오기 (없으면 동작 중단)"""
        p = self.params.get(item)
        if p is None:
            print("전달받은 메뉴가 없습니다.")
        return p

    def prepare_grip(self, item):
        """집기 전에 그리퍼 오픈"""
        p = self.get_params(item)
        if p is None:
            # 메뉴 파라미터 없으면 아무 동작도 하지 않음
            return
        
        open_width = p["open"]
        
        try:
            self.gripper.move_gripper(open_width)
            print(f"👐 [{item}] 그리퍼 오픈 {open_width}mm")
        except (AttributeError, TypeError):
            # 가상 그리퍼 혹은 단순 open()만 있는 경우
            try:
                self.gripper.open_gripper()
                print("   ⚠️ 그리퍼에 move_gripper(width) 메서드가 없어 open()으로 대체함")
            except Exception as e:
                print(f"   ❌ 그리퍼 오픈 실패: {e}")

    def execute_grip(self, item):
        """지정된 힘으로 물기"""
        p = self.get_params(item)
        if p is None:
            # 메뉴 파라미터 없으면 아무 동작도 하지 않음
            return

        force = p["force"]

        try:
            self.gripper.close_gripper(force)
            print(f"✊ [{item}] 그리퍼 클로즈(force={force})")
        except TypeError:
            # force 파라미터 없는 경우
            try:
                self.gripper.close_gripper()
                print("   ⚠️ 그리퍼에 close_gripper(force) 메서드가 없어 기본 close()로 대체함")
            except Exception as e:
                print(f"   ❌ 그리퍼 클로즈 실패: {e}")

    def release(self, item):
        """놓기: 아이템 종류는 필요없지만 통일성을 위해 포함"""
        p = self.get_params(item)
        if p is None:
            # 메뉴 파라미터 없으면 아무 동작도 하지 않음
            return

        open_width = p["open"]
        try:
            self.gripper.move_gripper(open_width)
            print("👐 오브젝트 해제")
        except (AttributeError, TypeError):
            try:
                self.gripper.open_gripper()
                print("   ⚠️ 그리퍼에 move_gripper(width) 없어서 open()으로 대체함")
            except Exception as e:
                print(f"   ❌ 그리퍼 오픈(릴리즈) 실패: {e}")