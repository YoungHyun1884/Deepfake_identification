import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # 이진 분류 (Real vs Fake)
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_frame(frame):
    img = Image.fromarray(frame)  # OpenCV 프레임을 PIL 이미지로 변환
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return "Fake" if pred == 1 else "Real"

# 비디오 파일 입력
video_path = "test_video.mp4"  # 예측할 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 예측
    label = predict_frame(frame)

    # 화면에 결과 출력
    color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
