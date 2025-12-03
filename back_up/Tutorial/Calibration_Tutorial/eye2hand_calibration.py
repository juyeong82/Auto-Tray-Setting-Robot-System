import json
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import os
from scipy.linalg import sqrtm
from numpy.linalg import inv

# 1) 로봇 그리퍼의 절대 좌표 (x, y, z, rx, ry, rz)를 행렬로 변환
def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    # 베이스->그리퍼 변환행렬 (4x4) 반환
    R = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# 2) 체커보드 코너 검출 (카메라->체커보드 변환 구하기)
def find_checkerboard_pose(image, board_size, square_size, camera_matrix, dist_coeffs):
    # board_size: 내부 코너 개수 (가로-1, 세로-1)
    # square_size: 격자 한 변의 길이 (mm)
    
    # 3D 기준점 생성
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 코너 찾기
    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None

    # 코너 좌표 정밀 보정
    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    # PnP로 카메라 기준 체커보드 자세 추정
    retval, rvec, tvec = cv2.solvePnP(objp, corners_sub, camera_matrix, dist_coeffs)
    if not retval:
        return None, None

    # 회전벡터 -> 회전행렬 변환
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

# 3) 체커보드 이미지를 이용한 카메라 내부 파라미터(Intrinsic) 보정
def calibrate_camera_from_chessboard(image_paths, board_size, square_size):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size

    obj_points = []  # 3D 점
    img_points = []  # 2D 점
    image_shape = None
    valid_image_count = 0

    print(f"-> Intrinsic Calibration 시작: 총 {len(image_paths)}장 처리 예정")

    for fname in image_paths:
        if not os.path.exists(fname):
            print(f"[경고] 파일 없음: {fname}")
            continue
            
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(objp)
            img_points.append(corners_sub)
            valid_image_count += 1
    
    print(f"-> 체커보드 인식 성공: {valid_image_count}장")

    if valid_image_count < 3:
        print("[에러] 체커보드를 인식한 이미지가 너무 적음 (최소 3장 이상 필요). board_size를 확인해볼 것.")
        return None, None, None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None
    )

    if not ret:
        print("[에러] Intrinsic Calibration 수렴 실패")
        return None, None, None, None

    return camera_matrix, dist_coeffs, rvecs, tvecs

# 4) 변환 행렬 리스트 병합
def compose_transformation_matrices(R_list, t_list):
    T_list = []
    for R, t in zip(R_list, t_list):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.ravel(t)
        T_list.append(T)
    return T_list

# 회전행렬 로그 변환
def logR(T):
    R = T[0:3, 0:3]
    # 회전각 계산 시 수치 오차로 -1~1 범위를 벗어나는 경우 클리핑
    trace_val = (np.trace(R) - 1) / 2
    trace_val = np.clip(trace_val, -1.0, 1.0)
    
    theta = np.arccos(trace_val)
    
    if np.isclose(theta, 0):
        return np.zeros(3)

    logr = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) * theta / (2 * np.sin(theta))
    return logr

# Park & Martin 방법으로 AX=XB 풀이
def Calibrate(A, B):
    n_data = len(A)
    print(f"-> Hand-Eye Calibration 계산 시작 (데이터 쌍: {n_data}개)")
    
    if n_data < 3:
        raise ValueError("데이터 부족: AX=XB를 풀기 위해 최소 3개 이상의 이동 데이터가 필요함.")

    M = np.zeros((3, 3))

    for i in range(n_data - 1):
        alpha  = logR(A[i])
        beta   = logR(B[i])
        alpha2 = logR(A[i + 1])
        beta2  = logR(B[i + 1])
        
        alpha3 = np.cross(alpha, alpha2)
        beta3  = np.cross(beta, beta2)
        
        M1 = np.dot(beta.reshape(3, 1), alpha.reshape(1, 3))
        M2 = np.dot(beta2.reshape(3, 1), alpha2.reshape(1, 3))
        M3 = np.dot(beta3.reshape(3, 1), alpha3.reshape(1, 3))
        
        M += M1 + M2 + M3

    # 회전 성분(theta) 계산
    # M이 0행렬이거나 특이행렬이면 여기서 에러 발생
    try:
        theta = np.dot(sqrtm(inv(np.dot(M.T, M))), M.T)
    except np.linalg.LinAlgError:
        print("[치명적 에러] M행렬이 특이행렬(Singular)임. 데이터가 너무 적거나 회전이 충분하지 않음.")
        raise

    # 이동 성분(b_x) 계산
    C = np.zeros((3 * n_data, 3))
    d = np.zeros((3 * n_data, 1))
    for i in range(n_data):
        rot_a = A[i][:3, :3]
        trans_a = A[i][:3, 3]
        trans_b = B[i][:3, 3]
        C[3 * i:3 * i + 3, :] = np.eye(3) - rot_a
        d[3 * i:3 * i + 3, 0] = trans_a - np.dot(theta, trans_b)
    
    b_x = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, d))
    return theta, b_x

# Main 실행
if __name__ == "__main__":
    # 데이터 로드
    try:
        data = json.load(open("data/calibrate_data.json"))
    except FileNotFoundError:
        print("[에러] json 파일을 찾을 수 없음. 경로 확인 필요.")
        exit()

    robot_poses = np.array(data["poses"])
    # 이미지 경로 절대 경로 변환 혹은 확인 필요 (현재는 상대경로 가정)
    image_paths = ["data/" + d for d in data["file_name"]]

    # 1. 로봇 포즈 유효성 검사 (특이점 확인)
    valid_indices = []
    for i, pose in enumerate(robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)
        det_T = np.linalg.det(T_base2gripper)
        if np.abs(det_T) > 1e-6:
            valid_indices.append(i)
        else:
            print(f"⚠️ Index {i}: 로봇 포즈 행렬이 특이행렬임 (제외됨)")

    robot_poses = robot_poses[valid_indices]
    image_paths = [image_paths[i] for i in valid_indices]

    # ==========================================
    # [중요] 체커보드 설정 확인 필수!
    # Inner Corner 개수 (가로 칸수 - 1, 세로 칸수 - 1)
    checkerboard_size = (10, 7)  # <-- 여기를 실제 보드에 맞게 수정 (예: 7, 5)
    square_size = 25            # mm 단위
    # ==========================================

    # 2. 카메라 내부 파라미터 캘리브레이션
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_chessboard(
        image_paths, checkerboard_size, square_size
    )

    if camera_matrix is None:
        print("[종료] 카메라 캘리브레이션 실패로 프로그램을 종료함.")
        exit()

    # 3. Hand-Eye 데이터 수집
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_checker2camera_list = []
    t_checker2camera_list = []

    print("-> 개별 이미지에 대한 Pose 추정 시작...")
    for img_path, pose in zip(image_paths, robot_poses):
        # Base -> Gripper
        T_base2gripper = get_robot_pose_matrix(*pose)
        
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Camera -> Checkerboard
        R_cam2checker, t_cam2checker = find_checkerboard_pose(
            image, checkerboard_size, square_size, camera_matrix, dist_coeffs
        )
        
        # 인식 실패 시 건너뜀
        if R_cam2checker is None:
            continue

        # Gripper -> Base (역행렬)
        T_gripper2base= np.linalg.inv(T_base2gripper)
        R_gripper2base_list.append(T_gripper2base[:3, :3].copy())
        t_gripper2base_list.append(T_gripper2base[:3, 3].reshape(-1, 1).copy())

        # Checkerboard -> Camera (역행렬)
        T_cam2checker = np.eye(4)
        T_cam2checker[:3, :3] = R_cam2checker
        T_cam2checker[:3, 3] = t_cam2checker.flatten()
        
        T_checker2cam = np.linalg.inv(T_cam2checker)
        R_checker2camera_list.append(T_checker2cam[:3, :3].copy())
        t_checker2camera_list.append(T_checker2cam[:3, 3].copy())

    # 4. 행렬 조합 및 A, B 행렬 생성
    T_gripper2base_list = compose_transformation_matrices(R_gripper2base_list, t_gripper2base_list)
    T_checker2cam_list = compose_transformation_matrices(R_checker2camera_list, t_checker2camera_list)
    
    A_list = []
    B_list = []
    num_pairs = min(len(T_gripper2base_list), len(T_checker2cam_list))
    
    print(f"-> 유효한 포즈 쌍 개수: {num_pairs}")

    if num_pairs < 3:
        print("[종료] 캘리브레이션을 위한 유효 데이터 쌍이 부족함 (최소 3개).")
        exit()

    # A: 로봇의 상대 움직임, B: 카메라의 상대 움직임
    for i in range(num_pairs - 1):
        A_i = np.dot(inv(T_gripper2base_list[i]), T_gripper2base_list[i + 1])
        B_i = np.dot(inv(T_checker2cam_list[i]), T_checker2cam_list[i + 1])
        A_list.append(A_i)
        B_list.append(B_i)

    # 5. 최종 캘리브레이션 수행
    try:
        theta, b_x = Calibrate(A_list, B_list)
        
        X = np.eye(4)
        X[:3, :3] = theta
        X[:3, 3] = b_x.flatten()
        
        T_cam2base = X
        print("\n=== 결과 (Camera -> Base) ===")
        print(T_cam2base)
        print("Translation Vector (xyz):", T_cam2base[:3, 3])
        
        np.save("T_cam2base.npy", T_cam2base)
        print("-> T_cam2base.npy 저장 완료")
        
    except Exception as e:
        print(f"[최종 실패] 캘리브레이션 계산 중 오류 발생: {e}")