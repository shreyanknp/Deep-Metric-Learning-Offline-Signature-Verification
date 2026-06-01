import io
from typing import Optional

import streamlit as st
from PIL import Image
import torch

from src.data.transforms import get_default_transforms
from src.models.backbones import SmallCNN, ResNet18Embed
from src.models.siamese import SiameseNetwork


@st.cache_resource
def load_model(arch: str = "small", emb_dim: int = 128, checkpoint_path: Optional[str] = None):
    if arch == "resnet18":
        encoder = ResNet18Embed(emb_dim=emb_dim)
    else:
        encoder = SmallCNN(emb_dim=emb_dim)
    model = SiameseNetwork(encoder=encoder)
    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
        except Exception:
            # try loading as encoder-only
            try:
                model.encoder.load_state_dict(state)
            except Exception:
                st.warning("Could not load checkpoint, using random weights.")
    model.eval()
    return model


def read_image(uploaded) -> Image.Image:
    if uploaded is None:
        return None
    data = uploaded.read()
    img = Image.open(io.BytesIO(data)).convert("L")
    return img


def embed_image(model, img: Image.Image, size: int = 224):
    transform = get_default_transforms(size)
    t = transform(img).unsqueeze(0)  # 1,C,H,W
    with torch.no_grad():
        emb = model.encoder(t)
    return emb.squeeze(0).numpy()


def main():
    st.title("Signature Comparison Demo")

    st.markdown("Upload two signature images to compare similarity.")

    col1, col2 = st.columns(2)
    with col1:
        up1 = st.file_uploader("Signature A", type=["png", "jpg", "jpeg"], key="a")
    with col2:
        up2 = st.file_uploader("Signature B", type=["png", "jpg", "jpeg"], key="b")

    arch = st.selectbox("Backbone", ["small", "resnet18"], index=0)
    emb_dim = st.number_input("Embedding dim", value=128, step=1)
    ckpt = st.text_input("Checkpoint path (optional)", value="")
    size = st.slider("Input size", min_value=64, max_value=512, value=224, step=32)

    model = load_model(arch=arch, emb_dim=emb_dim, checkpoint_path=ckpt or None)

    img1 = read_image(up1)
    img2 = read_image(up2)

    if img1 is not None:
        st.image(img1, caption="Signature A", width=300)
    if img2 is not None:
        st.image(img2, caption="Signature B", width=300)

    if st.button("Compare"):
        if img1 is None or img2 is None:
            st.error("Please upload two images.")
        else:
            emb1 = embed_image(model, img1, size=size)
            emb2 = embed_image(model, img2, size=size)
            import numpy as np

            dist = float(np.linalg.norm(emb1 - emb2))
            sim = 1.0 / (1.0 + dist)
            st.write(f"Distance: {dist:.4f}")
            st.write(f"Similarity score (1/(1+dist)): {sim:.4f}")

            thresh = st.slider("Decision threshold (distance)", min_value=0.0, max_value=10.0, value=1.0)
            label = "Genuine" if dist < thresh else "Forged"
            st.success(f"Result: {label}")


if __name__ == "__main__":
    main()
