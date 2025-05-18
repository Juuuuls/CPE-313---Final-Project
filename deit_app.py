import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# Update this to your actual class names
class_names = ['class0', 'class1']

# Path to your checkpoint
MODEL_PATH = "C:/Users/Ju/Downloads/weld_vit_split/vit_deit_classifier_95_40.pth"

# Preprocessing must match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

st.title("Weld Defect Classification with DeIT Vision Transformer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    input_tensor = transform(img).unsqueeze(0).to(device)

    model = load_model()
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = torch.argmax(outputs, 1).item()
        st.markdown(f"### Prediction: **{class_names[pred]}** (class index: {pred})")
