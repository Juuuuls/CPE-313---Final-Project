import streamlit as st
import torch
import timm
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np

# --- DeiT ViT classifier setup ---

class_names = ['bad weld', 'good weld']  # Adjust if your class names differ

MODEL_PATH_VIT = "C:/Users/Ju/Downloads/weld_vit_split/vit_deit_classifier_95_40.pth"

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_vit_model():
    model = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=2)
    state_dict = torch.load(MODEL_PATH_VIT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

# --- Mask R-CNN segmentation setup ---

MODEL_PATH_MASKRCNN = "C:/Users/Ju/Downloads/weld_vit_split/welding_defect_maskrcnn.pth"
num_classes_maskrcnn = 8  # Adjust according to your model's number of classes

label_map = {
    1: "Bad Welding",
    2: "Crack",
    3: "Excess Reinforcement",
    4: "Good Welding",
    5: "Porosity",
    6: "Spatters",
    7: "Weld"
}

@st.cache_resource
def load_maskrcnn_model():
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes_maskrcnn)
    model.load_state_dict(torch.load(MODEL_PATH_MASKRCNN, map_location=device))
    model.eval()
    model.to(device)
    return model

def visualize_predictions(image_tensor, outputs, label_map):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    masks = outputs['masks'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    boxes = outputs['boxes'].cpu().numpy()
    for i in range(len(masks)):
        if scores[i] < 0.5:
            continue
        mask = masks[i, 0]
        label = labels[i]
        score = scores[i]
        colored_mask = Image.fromarray((mask > 0.5).astype(np.uint8) * 128)
        img_pil.paste(colored_mask, (0, 0), colored_mask)
        pos = boxes[i]
        draw.rectangle([pos[0], pos[1], pos[2], pos[3]], outline="red", width=2)
        draw.text((pos[0], pos[1] - 10), f"{label_map.get(label, 'Unknown')}: {score:.2f}", fill="yellow")
    return img_pil

# --- Streamlit app ---

st.title("Weld Quality Classification and Defect Segmentation")

uploaded_file = st.file_uploader("Upload a welding image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load models
    vit_model = load_vit_model()
    maskrcnn_model = load_maskrcnn_model()

    # Preprocess for DeiT ViT classifier
    input_tensor_vit = transform_vit(img).unsqueeze(0).to(device)

    # Predict weld quality (optional display)
    with torch.no_grad():
        outputs_vit = vit_model(input_tensor_vit)
        pred_class = torch.argmax(outputs_vit, 1).item()#1
    st.markdown(f"### Weld Classification: **{class_names[pred_class]}**")

    # Always run Mask R-CNN segmentation
    img_tensor_maskrcnn = F.to_tensor(img).to(device)
    with torch.no_grad():
        outputs_maskrcnn = maskrcnn_model([img_tensor_maskrcnn])[0]

    # Filter predictions by confidence threshold
    conf_thresh = 0.5
    keep = outputs_maskrcnn['scores'] >= conf_thresh
    outputs_filtered = {k: v[keep] for k, v in outputs_maskrcnn.items()}

    # Visualize and show segmentation
    result_img = visualize_predictions(img_tensor_maskrcnn, outputs_filtered, label_map)
    st.image(result_img, caption="Segmentation Results", use_column_width=True)
