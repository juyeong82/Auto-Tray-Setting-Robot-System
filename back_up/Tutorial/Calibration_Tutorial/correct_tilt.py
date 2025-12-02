import numpy as np
from scipy.spatial.transform import Rotation

# =========================================================
# [설정] 3개의 지점에서 측정한 데이터를 입력하세요.
# P_robot: 로봇 팔을 실제로 이동시킨 좌표 (정답지)
# P_cam:   그 위치에서 카메라가 인식한 좌표 (오답지) -> test_view_only.py에서 '로봇 베이스 좌표'로 출력된 값
# =========================================================

# 1. 첫 번째 지점 (작업 영역의 중심 부근 권장)
P_robot_1 = [278.5, 231.7, 49.8]    # 예: 로봇을 (400, 0, 10)으로 보냄
P_cam_1   = [273.5, 231.7, 11.8]   # 예: 카메라로는 (405, -2, 12)로 나옴

# 2. 두 번째 지점 (X축 방향으로 멀리 이동한 곳)
P_robot_2 = [452.1, -85.2, 59.4]    # X를 200mm 더 이동
P_cam_2   = [427.1, -80.2, 4.4]    # 오차가 달라짐

# 3. 세 번째 지점 (Y축 방향으로 멀리 이동한 곳)
P_robot_3 = [619.5, -228.2, 64]  # Y를 200mm 더 이동
P_cam_3   = [584.5, -228.2, 9]  # 여기선 Z가 더 뜸
# =========================================================

def calibrate_3points():
    file_name = "T_cam2base.npy"
    
    try:
        # 1. 기존 행렬 로드
        T_current = np.load(file_name)
        print(f"📂 기존 행렬 로드 성공")

        # 2. 데이터 정리
        # 실제 로봇 위치 (Target)
        pts_robot = np.array([P_robot_1, P_robot_2, P_robot_3])
        # 카메라가 본 위치 (Source) -> 현재 T_current가 적용된 상태임
        pts_measured = np.array([P_cam_1, P_cam_2, P_cam_3])

        # 3. 오차 역산 (Kabsch Algorithm 응용)
        # 현재 카메라 측정값(pts_measured)을 로봇 정답(pts_robot)으로 매핑하는 변환 행렬 찾기
        
        # 중심점 계산 (Centroid)
        centroid_robot = np.mean(pts_robot, axis=0)
        centroid_measured = np.mean(pts_measured, axis=0)

        # 중심 기준으로 좌표 이동
        robot_centered = pts_robot - centroid_robot
        measured_centered = pts_measured - centroid_measured

        # 회전 행렬 계산 (SVD 방식)
        H = np.dot(measured_centered.T, robot_centered)
        U, S, Vt = np.linalg.svd(H)
        R_correction = np.dot(Vt.T, U.T)

        # 반사(Reflection) 확인 및 수정
        if np.linalg.det(R_correction) < 0:
            Vt[2, :] *= -1
            R_correction = np.dot(Vt.T, U.T)

        # 이동(Translation) 계산
        t_correction = centroid_robot - np.dot(R_correction, centroid_measured)

        # 4. 보정 행렬 구성 (Error Correction Matrix)
        T_error = np.eye(4)
        T_error[:3, :3] = R_correction
        T_error[:3, 3] = t_correction

        # 5. 최종 행렬 업데이트
        # 새로운 T = T_error * T_current
        # (기존 변환 결과에 오차 보정을 추가로 적용)
        T_new = np.dot(T_error, T_current)

        # 6. 결과 출력
        r = Rotation.from_matrix(R_correction)
        euler = r.as_euler('xyz', degrees=True)
        print("\n📊 [분석 결과]")
        print(f" -> 추가 보정된 회전(Roll, Pitch, Yaw): {euler}")
        print(f" -> 추가 보정된 이동(XYZ): {t_correction}")
        
        print("\n--- [검증: 보정 후 예상 좌표] ---")
        for i, (p_meas, p_rob) in enumerate(zip(pts_measured, pts_robot)):
            # 검증 수식: (R_corr * P_meas) + t_corr
            p_corrected = np.dot(R_correction, p_meas) + t_correction
            err = np.linalg.norm(p_corrected - p_rob)
            print(f"점 {i+1}: 목표{p_rob} vs 예측{np.round(p_corrected, 2)} (오차: {err:.2f}mm)")

        # 저장
        np.save(file_name, T_new)
        print(f"\n✅ '{file_name}' 업데이트 완료!")

    except FileNotFoundError:
        print("🚨 파일이 없습니다.")

if __name__ == "__main__":
    calibrate_3points()