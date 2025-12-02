import cv2

def scan_cameras(max_index=10):
    print(f"Searching camera indices (0 ~ {max_index - 1})...")
    
    available_indices = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        
        # 카메라 장치 연결 시도
        if cap.isOpened():
            # 확실한 확인을 위해 프레임 읽기 시도
            ret, _ = cap.read()
            if ret:
                print(f"✅ Camera found at index: {index}")
                available_indices.append(index)
            else:
                print(f"⚠️ Index {index} is open but returning no frames (Possibly generic driver)")
            cap.release()
        
    print("-" * 30)
    print(f"Result: Available Indices -> {available_indices}")

if __name__ == "__main__":
    scan_cameras()