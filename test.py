model.load_state_dict(torch.load("deepfake_detector.pth"))
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return "Fake" if pred == 1 else "Real"

# 테스트
test_image = "dataset/frames/fake/frame_0001.jpg"
print(f"Prediction: {predict(test_image)}")
