from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, utils as vutils
from PIL import Image
import os
import uuid

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= CLASSIFIER =================

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, 128),   # 128x128 input
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = CNN().to(device)
model.load_state_dict(torch.load("../model/saved_model.pt", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class_names = ["Healthy", "Late_blight", "Leaf_Mold"]

# ================= GAN GENERATOR (128x128) =================

class Generator(nn.Module):
    def __init__(self, nz=100):
        super().__init__()
        self.main = nn.Sequential(

            # 1x1 → 4x4
            nn.ConvTranspose2d(nz, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 4 → 8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8 → 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16 → 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32 → 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 → 128
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


generator = Generator().to(device)
generator.load_state_dict(torch.load("../model/tomato_late_blight_gan.pt", map_location=device))
generator.eval()

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return jsonify({
            "prediction": class_names[predicted.item()],
            "confidence": round(confidence.item() * 100, 2)
        })

    except Exception as e:
        print("PREDICT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["GET"])
def generate_image():

    noise = torch.randn(1, 100, 1, 1, device=device)

    with torch.no_grad():
        fake_image = generator(noise).detach().cpu()

    os.makedirs("static/generated", exist_ok=True)

    filename = str(uuid.uuid4()) + ".png"
    save_path = os.path.join("static/generated", filename)

    vutils.save_image(
        fake_image,
        save_path,
        normalize=True,
        value_range=(-1, 1)
    )

    return jsonify({
        "image_url": "/static/generated/" + filename
    })


if __name__ == "__main__":
    app.run(debug=True)