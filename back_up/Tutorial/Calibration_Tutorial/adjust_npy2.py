import numpy as np
from scipy.spatial.transform import Rotation

# =========================================================
# [데이터 입력 완료] 사용자가 제공한 13개 포인트 데이터
# 형식: [ [카메라가_측정한_Base_XYZ], [실제_로봇_XYZ] ]
# =========================================================
data_pairs = [
    # 포인트 1
    # Cam: X=713.7, Y=-22.2, Z=21.4
    # Real: X=690.042, Y=11.454, Z=13.122
    [[713.7, -22.2, 21.4], [690.042, 11.454, 13.122]],

    # 포인트 2
    # Cam: X=704.4, Y=163.2, Z=11.0
    # Real: X=675.017, Y=177.038, Z=12.656
    [[704.4, 163.2, 11.0], [675.017, 177.038, 12.656]],

    # 포인트 3
    # Cam: X=450.4, Y=215.0, Z=4.5
    # Real: X=422.919, Y=225.018, Z=11.38
    [[450.4, 215.0, 4.5], [422.919, 225.018, 11.38]],

    # 포인트 4 (첫 번째)
    # Cam: X=294.6, Y=233.9, Z=-6.4
    # Real: X=274.818, Y=224.649, Z=10.43
    [[294.6, 233.9, -6.4], [274.818, 224.649, 10.43]],

    # 포인트 4 (두 번째 - 중간 지점)
    # Cam: X=448.1, Y=66.7, Z=1.2
    # Real: X=424.632, Y=76.843, Z=10.189
    [[448.1, 66.7, 1.2], [424.632, 76.843, 10.189]],

    # 포인트 5
    # Cam: X=599.3, Y=48.1, Z=17.1
    # Real: X=575.896, Y=77.037, Z=11.739
    [[599.3, 48.1, 17.1], [575.896, 77.037, 11.739]],

    # 포인트 6
    # Cam: X=298.5, Y=-95.7, Z=2.5
    # Real: X=274.987, Y=-74.174, Z=10.532
    [[298.5, -95.7, 2.5], [274.987, -74.174, 10.532]],

    # 포인트 7
    # Cam: X=294.9, Y=-243.2, Z=-0.3
    # Real: X=277.731, Y=-222.733, Z=10.656
    [[294.9, -243.2, -0.3], [277.731, -222.733, 10.656]],

    # 포인트 8
    # Cam: X=595.7, Y=-247.8, Z=12.3
    # Real: X=575.705, Y=-222.704, Z=10.814
    [[595.7, -247.8, 12.3], [575.705, -222.704, 10.814]],

    # 포인트 9
    # Cam: X=521.8, Y=-417.8, Z=49.2
    # Real: X=505.679, Y=-390.183, Z=46.873
    [[521.8, -417.8, 49.2], [505.679, -390.183, 46.873]],

    # 포인트 10
    # Cam: X=524.4, Y=-509.8, Z=43.9
    # Real: X=502.585, Y=-474.897, Z=45.0
    [[524.4, -509.8, 43.9], [502.585, -474.897, 45.0]],

    # 포인트 11
    # Cam: X=284.8, Y=-515.2, Z=36.6
    # Real: X=273.89, Y=-484.603, Z=46.578
    [[284.8, -515.2, 36.6], [273.89, -484.603, 46.578]],

    # 포인트 12
    # Cam: X=290.7, Y=-416.0, Z=38.2
    # Real: X=273.818, Y=-390.886, Z=46.001
    [[290.7, -416.0, 38.2], [273.818, -390.886, 46.001]]
]
# =========================================================

def calibrate_fine_tune_final():
    # file_name = "T_cam2base.npy"
    # 경로가 다르면 아래와 같이 절대 경로를 사용하세요
    file_name = "/home/juyeong/ros2_ws/src/back_up/Tutorial/Calibration_Tutorial/T_cam2base.npy"
    
    try:
        # 1. 기존 행렬 로드
        T_current = np.load(file_name)
        print(f"📂 '{file_name}' 로드 성공")
        print("--- [보정 전] ---")

        # 2. 데이터 분리
        pts_measured = np.array([pair[0] for pair in data_pairs]) # 카메라 측정값 (Source)
        pts_real     = np.array([pair[1] for pair in data_pairs]) # 실제 로봇값 (Target)

        # 3. 최적 변환 행렬 계산 (SVD 방식)
        centroid_measured = np.mean(pts_measured, axis=0)
        centroid_real = np.mean(pts_real, axis=0)

        measured_centered = pts_measured - centroid_measured
        real_centered = pts_real - centroid_real

        H = np.dot(measured_centered.T, real_centered)
        U, S, Vt = np.linalg.svd(H)
        R_fix = np.dot(Vt.T, U.T)

        if np.linalg.det(R_fix) < 0:
            Vt[2, :] *= -1
            R_fix = np.dot(Vt.T, U.T)

        t_fix = centroid_real - np.dot(R_fix, centroid_measured)

        # 4. 보정 행렬 구성
        T_fix = np.eye(4)
        T_fix[:3, :3] = R_fix
        T_fix[:3, 3] = t_fix

        # 5. 최종 행렬 업데이트
        T_new = np.dot(T_fix, T_current)

        # 6. 결과 리포트
        r = Rotation.from_matrix(R_fix)
        euler = r.as_euler('xyz', degrees=True)
        print("\n📊 [보정 결과 분석]")
        print(f" -> 추가 회전 보정 (Roll, Pitch, Yaw): {np.round(euler, 4)}")
        print(f" -> 추가 위치 보정 (X, Y, Z): {np.round(t_fix, 4)}")
        
        print("\n--- [검증: 보정 후 예상 오차] ---")
        total_error = 0
        for i, (p_meas, p_real) in enumerate(zip(pts_measured, pts_real)):
            p_corrected = np.dot(R_fix, p_meas) + t_fix
            err_xyz = p_corrected - p_real
            err_dist = np.linalg.norm(err_xyz)
            total_error += err_dist
            print(f"점 {i+1}: 오차 {err_dist:.2f}mm (잔여차이: X{err_xyz[0]:.1f}, Y{err_xyz[1]:.1f}, Z{err_xyz[2]:.1f})")

        print(f"\n✅ 평균 잔여 오차: {total_error / len(data_pairs):.2f}mm")

        # 7. 저장
        np.save(file_name, T_new)
        print(f"\n💾 '{file_name}' 파일이 성공적으로 업데이트되었습니다!")
        print("-> 이제 다시 측정 코드를 실행하여 좌표가 일치하는지 확인하세요.")

    except FileNotFoundError:
        print(f"🚨 '{file_name}' 파일이 없습니다. 경로를 확인해주세요.")

if __name__ == "__main__":
    calibrate_fine_tune_final()