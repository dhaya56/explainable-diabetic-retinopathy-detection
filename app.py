# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.ndimage import gaussian_filter
import cv2

# -------------------------------
# 1️⃣ CNN Model (Your Architecture)
# -------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        self.flatten_dim = self.get_flatten_dim()

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def get_flatten_dim(self):
        x = torch.zeros(1, 3, 512, 512)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------------------
# 2️⃣ Grad-CAM Utility Functions
# -------------------------------
def get_last_conv_layer(model):
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


def preprocess_image(pil_img, img_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    return transform(pil_img).unsqueeze(0)


def compute_gradcam(model, input_tensor, target_class=None):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    target_layer = get_last_conv_layer(model)
    activations, gradients = [], []

    def forward_hook(module, inp, outp):
        activations.append(outp.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    if target_class is None:
        target_class = int(torch.argmax(probs, dim=1).item())

    score = outputs[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    weights = torch.mean(grad.view(grad.size(0), -1), dim=1)

    cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam.cpu().numpy(), 0)
    fh.remove()
    bh.remove()
    return cam, probs.detach().cpu().numpy()[0]


def postprocess_cam(cam, img_size):
    cam = cam / cam.max() if cam.max() != 0 else cam
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize(img_size, resample=Image.BILINEAR)
    return np.array(cam)


# -------------------------------
# 3️⃣ Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Diabetic Retinopathy Detection — Grad-CAM", layout="wide")
st.title("🩺 Diabetic Retinopathy Detection with Grad-CAM")
st.write("Upload a retinal image to visualize affected regions and get AI-based medical guidance.")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model path", "model.pth")
classes_path = st.sidebar.text_input("Classes path", "classes.pth")
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.5)
apply_smoothing = st.sidebar.checkbox("Smooth Heatmap", value=True)
threshold_ratio = st.sidebar.slider("Detection Threshold (%)", 10, 100, 70, step=5) / 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    class_names = torch.load(classes_path)
    num_classes = len(class_names)
except Exception as e:
    st.error(f"Error loading classes: {e}")
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]
    num_classes = len(class_names)

model = CNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

uploaded_file = st.file_uploader("📤 Upload Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    input_tensor = preprocess_image(image, img_size=(512, 512))
    cam_raw, probs = compute_gradcam(model, input_tensor)

    if apply_smoothing:
        cam_raw = gaussian_filter(cam_raw, sigma=3)

    probs_tensor = torch.tensor(probs)
    top_idx = int(torch.argmax(probs_tensor).item())
    predicted_label = class_names[top_idx]
    confidence = probs_tensor[top_idx].item()

    # -------------------------------
    # ✅ Improved Grad-CAM Overlay (Removes Black Background Artifacts)
    # -------------------------------
    cam_uint8 = postprocess_cam(cam_raw, image.size)

    # Convert images to numpy for masking
    image_np = np.array(image)
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Create a binary mask (ignore black background)
    _, mask = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask / 255.0

    # Apply mask to Grad-CAM heatmap
    cam_masked = np.uint8(cam_uint8 * mask)

    # Optional: threshold weak activations to clean up
    cam_masked[cam_masked < 0.2 * cam_masked.max()] = 0

    # Colorize Grad-CAM
    cmap = plt.get_cmap("jet")
    cam_colored = cmap(cam_masked / 255.0)[:, :, :3]

    # Blend Grad-CAM heatmap with original image
    cam_overlay = Image.blend(
        image.convert("RGBA"),
        Image.fromarray((cam_colored * 255).astype(np.uint8)).convert("RGBA"),
        alpha=alpha
    )

    # Detect most active region (to draw circle)
    gray_cam = cam_masked.copy()
    max_val = gray_cam.max()
    threshold = threshold_ratio * max_val
    y_max, x_max = np.unravel_index(np.argmax(gray_cam), gray_cam.shape)
    max_intensity = gray_cam[y_max, x_max]

    overlay_cv = np.array(cam_overlay.convert("RGB"))

    # Draw circle only if DR region is confidently detected
    if predicted_label.lower() not in ["no_dr", "no dr", "normal"] and max_intensity >= threshold:
        radius = int(min(overlay_cv.shape[:2]) * 0.04)
        cv2.circle(overlay_cv, (x_max, y_max), radius, (255, 0, 0), 3)
        cv2.putText(
            overlay_cv,
            "Affected Region",
            (x_max + 10, y_max - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    cam_overlay = Image.fromarray(overlay_cv)

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("🔥 Grad-CAM Visualization")
        st.image(cam_overlay, use_container_width=True)

    st.subheader("🔍 Model Prediction")
    st.info(f"Predicted: **{predicted_label} ({confidence*100:.2f}%)**")

    # -------------------------------
    # ✅ DIABETIC RETINOPATHY GUIDANCE BASED ON PREDICTION
    # -------------------------------
    st.subheader("📌 AI Health Guidance")

    if predicted_label.lower() in ["no_dr", "no dr", "normal"]:
        st.success("""
🟢 **No Diabetic Retinopathy Detected**

✅ Keep doing:
- Maintain normal blood glucose
- Yearly eye check-up
- Balanced diet, avoid junk food

🥗 **Good foods:** spinach, broccoli, almonds, carrots, berries  
⚠️ *Even with No-DR, regular screening is necessary.*
""")

    elif predicted_label.lower() == "mild":
        st.warning("""
🟡 **Mild Diabetic Retinopathy**

📌 **Precautions**
- Strict sugar control
- Regular BP & cholesterol check
- Avoid smoking/alcohol
- Avoid high-sugar & fried foods

🥗 **Recommended Foods**
- Leafy vegetables, citrus fruits, nuts, olive oil, fish
- Whole grains, beans, lentils

👁️ **Next Step**
- Eye check-up every 6 months
""")

    elif predicted_label.lower() == "moderate":
        st.warning("""
🟠 **Moderate Diabetic Retinopathy**

📌 **What To Do**
- Keep HbA1c < 7%
- Monitor BP, cholesterol
- Avoid processed & salty foods

🥗 **Eat**
- Spinach, kale, carrots, oranges, berries
- Walnuts, flaxseeds, oats, brown rice

💊 **Possible Advice**
- Early laser or anti-VEGF may be suggested by doctor

👁️ **Next Step**
- Visit ophthalmologist every 3–4 months
""")

    elif predicted_label.lower() == "severe":
        st.error("""
🔴 **Severe Diabetic Retinopathy — High Risk**

⚠️ **Immediate Action Needed**
- Visit eye specialist ASAP
- Likely need laser or anti-VEGF treatment

🥗 **Recommended Foods**
- Berries, leafy greens, carrots
- Fish, walnuts, flaxseed, oats

💊 **Possible Medications**
- Anti-VEGF (Ranibizumab, Aflibercept)
- Steroid eye injections (under supervision)

❌ **Avoid**
- Sugary drinks, junk food, smoking
""")

    elif any(x in predicted_label.lower() for x in ["proliferative", "proliferate_dr", "proliferatedr"]):
        st.error("""
🚨 **Proliferative Diabetic Retinopathy — Emergency**

⚠️ **Urgent Medical Care Required**
- High chance of vision loss
- Immediate retina specialist visit
- May require laser surgery or vitrectomy

🥗 **Supportive Foods**
- Leafy greens, carrots, berries
- Fish / walnuts / flax for omega-3
- Whole grains & legumes

💊 **Doctor May Suggest**
- Laser photocoagulation
- Anti-VEGF injections
- Vitrectomy

❗ **Do NOT Delay Treatment**
""")

    st.caption("⚠️ *AI guidance is for educational purposes only. Always consult an ophthalmologist for clinical advice.*")

    # Download overlay
    buf = io.BytesIO()
    cam_overlay.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        "📥 Download Grad-CAM Overlay",
        data=buf,
        file_name="gradcam_overlay.jpg",
        mime="image/jpeg"
    )
