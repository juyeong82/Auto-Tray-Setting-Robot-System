from ultralytics import YOLO
import cv2
import os

# 파일 경로 설정 (절대 경로가 지정되어 있으므로 바로 사용)
model_filename = "/home/juyeong/ros2_ws/src/back_up/yolo_OD/best_seg.pt"

# 모델 로드
print(f"모델을 로드 중입니다: {model_filename}")
try:
    model = YOLO(model_filename)
except Exception as e:
    print(f"모델 로드 실패: {e}")
    exit()

# 카메라 열기 (0: 기본 내장 카메라, 6: 사용자 설정)
cap = cv2.VideoCapture(6)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 세그멘테이션 예측 시작 (종료: Q 키 누르기)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임 예측 (stream=False: 프레임 1개씩 예측)
    # 세그멘테이션을 위해 retina_masks=True 옵션을 주면 마스크 품질이 더 좋아집니다.
    results = model.predict(source=frame, conf=0.25, verbose=False, retina_masks=True)

    # === [수정된 부분] ===
    # results[0].plot() 함수는 바운딩 박스와 '세그멘테이션 마스크'를
    # 원본 이미지 위에 합성해서 반환해줍니다.
    annotated_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("YOLO Segmentation Predict", annotated_frame)

    # Q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
print("예측 종료")