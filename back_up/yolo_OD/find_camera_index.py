import cv2

def check_camera_indices():
    # 0번부터 20번까지 빠르게 스캔
    for index in range(20):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[성공] 카메라 인덱스 발견: {index}")
            else:
                print(f"[경고] 인덱스 {index}는 열리지만 프레임 캡처 실패 (Depth/IR 노드일 가능성)")
            cap.release()
        else:
            # 보통 연결 안 된 포트는 그냥 넘어감
            pass

if __name__ == "__main__":
    print("카메라 인덱스 스캔 시작...")
    check_camera_indices()
    print("스캔 종료.")