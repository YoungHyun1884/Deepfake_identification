import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

# 📌 데이터 로더 설정
class DeepfakeDataset(Dataset):
    def __init__(self, real_path, fake_path, transform=None):
        self.real_images = glob.glob(real_path + "/*.jpg")
        self.fake_images = glob.glob(fake_path + "/*.jpg")
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.images = self.real_images + self.fake_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# 📌 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 📌 데이터 로더 생성
real_data_path = "/home/abc/Desktop/딥페이크/dataset/frames/real"
fake_data_path = "/home/abc/Desktop/딥페이크/dataset/frames/fake"

dataset = DeepfakeDataset(real_path=real_data_path, fake_path=fake_data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("✅ 데이터 로딩 완료! 학습 시작")

# 📌 EfficientNet 모델 정의 (Warning 해결)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 이진 분류 (진짜 vs 가짜)
model = model.to(device)

# 📌 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 📌 모델 학습
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

# 📌 모델 저장
torch.save(model.state_dict(), "/home/abc/Desktop/딥페이크/model.pth")
print("🎉 모델 학습 완료 및 저장 완료!")
