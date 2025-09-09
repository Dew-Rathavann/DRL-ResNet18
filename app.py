# file: waste_classifier_app.py
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Define model architecture that matches checkpoint naming
class WasteClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(WasteClassifier, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class WasteClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waste Classification App")
        self.image_path = None

        self.label = QLabel("Upload an image of waste", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.clear_image)
        self.cancel_btn.setEnabled(False)

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict_image)
        self.predict_btn.setEnabled(False)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setEnabled(False)

        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.predict_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_label)
        layout.addWidget(self.clear_btn)

        self.setLayout(layout)
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['Electronic Waste', 'Organic Waste', 'Recycle Waste', 'Trash']

    def load_model(self):
        model = WasteClassifier(num_classes=4)
        checkpoint = torch.load("best_model_weights.pth", map_location=torch.device('cpu'))
        state_dict = checkpoint.get("resnet", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            self.image_label.setPixmap(QPixmap(file_path).scaled(300, 300, Qt.KeepAspectRatio))
            self.cancel_btn.setEnabled(True)
            self.predict_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)

    def clear_image(self):
        self.image_label.clear()
        self.image_path = None
        self.cancel_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.result_label.setText("")

    def predict_image(self):
        if not self.image_path:
            return
        image = Image.open(self.image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.classes[predicted.item()]
        self.result_label.setText(f"Predicted class: {predicted_class}")

    def clear_all(self):
        self.clear_image()
        self.result_label.setText("")
        self.clear_btn.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WasteClassifierApp()
    window.resize(400, 600)
    window.show()
    sys.exit(app.exec_())