import numpy as np
from scipy.spatial.transform import Rotation

def get_tf_from_npy():
    try:
        # 1. 캘리브레이션 결과 로드
        T = np.load("/home/juyeong/ros2_ws/src/back_up/Tutorial/Calibration_Tutorial/T_cam2base.npy")
        
        # 2. 이동 (Translation)
        x, y, z = T[:3, 3]
        
        # 3. 회전 (Rotation Matrix -> Quaternion)
        # ROS TF는 Quaternion (x, y, z, w)를 사용함
        r = Rotation.from_matrix(T[:3, :3])
        qx, qy, qz, qw = r.as_quat()
        
        print("\n" + "="*50)
        print("✅ [TF 등록용 명령어 arguments]")
        print("static_transform_publisher를 실행할 때 아래 숫자를 복사해 쓰세요.")
        print("-" * 50)
        # x y z qx qy qz qw frame_id child_frame_id
        # 보통 두산 로봇 베이스는 'base_0' 또는 'world', 카메라는 'camera_link'
        print(f"--x {x/1000.0:.5f} --y {y/1000.0:.5f} --z {z/1000.0:.5f} "
              f"--qx {qx:.5f} --qy {qy:.5f} --qz {qz:.5f} --qw {qw:.5f}")
        print("-" * 50)
        print("⚠️ 주의: 위 값은 미터(m) 단위로 변환되었습니다. (ROS 표준)")
        print("="*50)

    except FileNotFoundError:
        print("🚨 T_cam2base.npy 파일이 없습니다.")

if __name__ == "__main__":
    get_tf_from_npy()