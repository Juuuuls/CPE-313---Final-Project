import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
    def forward(self, x):
        norm_x = self.norm1(x)
        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x
        norm_x = self.norm2(x)
        x = self.mlp(norm_x) + x
        return x

def extract_patches(image_tensor, patch_size=4):
    bs, c, h, w = image_tensor.size()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    return unfolded

class ViT(nn.Module):
    def __init__(self, image_size, channels_in, patch_size, hidden_size, num_layers, num_heads=8):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, 10)  # 10 classes as per your training
        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_size))
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.001))

    def forward(self, image):
        bs = image.shape[0]
        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)
        patch_emb = patch_emb + self.pos_embedding
        embs = torch.cat((self.out_vec.expand(bs, 1, -1), patch_emb), 1)
        for block in self.blocks:
            embs = block(embs)
        return self.fc_out(embs[:, 0])

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = ViT(
        image_size=224,
        channels_in=3,
        patch_size=4,
        hidden_size=64,
        num_layers=4,
        num_heads=4
    )
    model_path = r"C:\Users\Ju\Downloads\weld_vit_split\vit_weld_classifier_90_37.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

st.title("Weld Defect Classification with Vanilla Vision Transformer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    
    st.write(f"Predicted class: {predicted_class.item()}")
    st.write(f"Confidence: {confidence.item():.2%}")
