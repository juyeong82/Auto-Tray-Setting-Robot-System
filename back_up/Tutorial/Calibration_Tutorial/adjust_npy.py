import numpy as np

# ==========================================
# [설정] 찾아낸 오차값 입력 (단위: mm)
# 테스트 코드에서 썼던 OFFSET 값을 그대로 넣으세요.
OFFSET_X = 0  
OFFSET_Y = 0
OFFSET_Z = -40.0    # 높이는 잘 맞으면 0
# ==========================================

def apply_offset():
    try:
        # 1. 기존 행렬 로드
        T_old = np.load("T_cam2base.npy")
        print("📂 기존 행렬 로드 성공")
        print("--- 수정 전 위치 (Translation) ---")
        print(f"X: {T_old[0, 3]:.2f}, Y: {T_old[1, 3]:.2f}, Z: {T_old[2, 3]:.2f}")

        # 2. 오차 적용 (Base Frame 기준 이동)
        # 행렬의 4번째 열(Translation Vector)에 오차를 더합니다.
        T_new = T_old.copy()
        T_new[0, 3] += OFFSET_X
        T_new[1, 3] += OFFSET_Y
        T_new[2, 3] += OFFSET_Z

        print("\n⚡ 오차 보정 적용 중...")
        print(f" -> X축: {OFFSET_X} mm 추가")
        print(f" -> Y축: {OFFSET_Y} mm 추가")

        print("\n--- 수정 후 위치 (Translation) ---")
        print(f"X: {T_new[0, 3]:.2f}, Y: {T_new[1, 3]:.2f}, Z: {T_new[2, 3]:.2f}")

        # 3. 저장
        np.save("T_cam2base.npy", T_new) # 덮어쓰기 (불안하면 이름 바꾸세요)
        print("\n✅ 'T_cam2base.npy' 파일이 보정된 값으로 저장되었습니다!")
        print("이제 test 코드를 실행할 때 OFFSET을 0으로 하고 돌려보세요.")

    except FileNotFoundError:
        print("🚨 T_cam2base.npy 파일이 없습니다.")

if __name__ == "__main__":
    apply_offset()