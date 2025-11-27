from ultralytics import YOLO
from datetime import datetime
import os

project_folder = "cvs"
model_filename = "best.pt" # train 디렉토리에 있는 best.pt를 project_folder로 복사
data_yaml_filename = "data.yaml"
predict_images_path = "predict_images"

# 자동으로 현재 폴더 기준 경로 설정
BASE_DIR = os.getcwd()  # 현재 폴더 기준
OUTPUT_DIR = os.path.join(BASE_DIR, project_folder)  # 원하는 상위 폴더 이름

# 날짜/시간 기반 하위 폴더 이름 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
predict_name = f"predict_{timestamp}"

model_path = os.path.join(OUTPUT_DIR, model_filename)
data_yaml_path = os.path.join(OUTPUT_DIR, data_yaml_filename)

# Model 로드
model = YOLO(model_path)

# 예측 실행
results = model.predict(
    source=os.path.join(OUTPUT_DIR, predict_images_path),  # 예측 대상 이미지 폴더
    conf=0.25,
    project=OUTPUT_DIR,
    name=predict_name,
    save=True,
    save_txt=True
)

# 최종 출력 경로 확인
print(f"Predict : {os.path.join(OUTPUT_DIR, predict_name)}")

##추가: 바운딩 박스 좌표 받기
# 2. 결과에서 좌표(Bounding Box) 추출하기
for result in results:
    # 현재 처리 중인 이미지 경로
    img_path = result.path
    filename = os.path.basename(img_path)
    print(f"Image: {filename}")

    # 감지된 모든 박스 정보 가져오기
    boxes = result.boxes

    # 박스가 하나도 없을 경우 처리
    if len(boxes) == 0:
        print("  - No objects detected.")
        continue

    # 각 박스(객체)별로 정보 추출
    for box in boxes:
        # (1) 좌표 추출 (xyxy: x1, y1, x2, y2)
        # cpu()로 옮기고 numpy로 변환해야 파이썬에서 쓰기 편함
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() 
        
        # (2) 클래스 ID 및 이름
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        
        # (3) 신뢰도 (Confidence)
        confidence = box.conf[0].item()

        print(f"  - Detected: {class_name} ({confidence:.2f})")
        print(f"    Coords (x1, y1, x2, y2): {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")
        
        # 💡 로봇 제어용 팁: 중심 좌표가 필요하다면?
        # x_center, y_center, width, height = box.xywh[0].cpu().numpy()
        # print(f"    Center: ({x_center}, {y_center})")

print("-" * 50)
