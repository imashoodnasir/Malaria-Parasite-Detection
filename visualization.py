import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from hybrid_capnet import HybridCapNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output, _ = self.model(input_tensor)
        loss = output[:, class_idx].sum()
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return cam

def show_cam_on_image(img_path, model_path, target_layer="backbone.layer3"):
    model = HybridCapNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    cam_extractor = GradCAM(model, target_layer)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    raw_img = Image.open(img_path).convert("RGB")
    input_tensor = tf(raw_img).unsqueeze(0).to(device)
    pred = model(input_tensor)[0]
    pred_idx = pred.argmax(dim=1).item()
    cam = cam_extractor.generate(input_tensor, pred_idx)
    img_np = np.array(raw_img.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + img_np
    cam_img = cam_img / np.max(cam_img)
    plt.imshow(cam_img)
    plt.axis('off')
    plt.title(f"Class: {pred_idx}")
    plt.show()

show_cam_on_image("sample.jpg", "best_model.pth")
