import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# 파일 로드
filename = "T_cam2base.npy"
if not os.path.exists(filename):
    print(f"❌ '{filename}' 파일이 없습니다. 경로를 확인하세요.")
    exit()

T_org = np.load(filename)

# 두산 로봇 기준 프레임 (보통 base_0 사용)
# 만약 RViz에서 로봇 베이스가 'link_base'나 'world'라면 그걸로 수정하세요.
PARENT_FRAME = "base_0" 
CHILD_FRAME = "camera_depth_optical_frame"

def print_tf_command(T, name):
    # 단위 변환 (mm -> m)
    x = T[0, 3] / 1000.0
    y = T[1, 3] / 1000.0
    z = T[2, 3] / 1000.0
    
    # 회전 변환
    quat = R.from_matrix(T[:3, :3]).as_quat() # x, y, z, w
    
    cmd = (
        f"ros2 run tf2_ros static_transform_publisher "
        f"--x {x:.5f} --y {y:.5f} --z {z:.5f} "
        f"--qx {quat[0]:.5f} --qy {quat[1]:.5f} --qz {quat[2]:.5f} --qw {quat[3]:.5f} "
        f"--frame-id {PARENT_FRAME} --child-frame-id {CHILD_FRAME}"
    )
    
    print(f"\n🔹 [{name}] 명령어:")
    print("-" * 20)
    print(cmd)
    print("-" * 20)

print("\n🔍 분석 결과")
print("=" * 60)

# 1. 원래 행렬 그대로 사용
print_tf_command(T_org, "옵션 1: 원래 행렬 (Original)")

# 2. 역행렬 사용 (이게 정답일 확률 90%)
T_inv = np.linalg.inv(T_org)
print_tf_command(T_inv, "옵션 2: 역행렬 (Inverse) - ⭐추천⭐")

print("=" * 60)
print("👉 [Tip] '옵션 2'를 먼저 복사해서 실행해보세요.")
print("👉 그래도 이상하면 '--frame-id'를 'base_0' 대신 'base_link'로 바꿔보세요.")