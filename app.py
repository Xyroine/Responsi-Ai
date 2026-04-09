import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Animal Face Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0c0c0f;
    color: #e8e6e0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 760px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-tag {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #a08c6e;
    border: 1px solid #2e2c28;
    border-radius: 100px;
    padding: 4px 14px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: #f0ede6;
    margin: 0 0 16px;
}
.hero-title span {
    background: linear-gradient(135deg, #d4a96a 0%, #e8c99a 50%, #c28a45 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px;
    font-weight: 300;
    color: #7a786f;
    line-height: 1.7;
    max-width: 420px;
    margin: 0 auto;
}

/* ── Upload card ── */
.upload-section {
    background: #141418;
    border: 1px solid #222228;
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
}

/* ── Streamlit uploader override ── */
[data-testid="stFileUploadDropzone"] {
    background: #0c0c0f !important;
    border: 2px dashed #2e2c28 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #a08c6e !important;
}
[data-testid="stFileUploadDropzone"] label {
    color: #5a5850 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Preview image ── */
.image-frame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #222228;
    margin: 1rem 0;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c28a45 0%, #d4a96a 100%) !important;
    color: #0c0c0f !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    height: auto !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 1rem;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(194, 138, 69, 0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    background: #141418;
    border: 1px solid #222228;
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5850;
    margin-bottom: 10px;
}
.result-animal {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #d4a96a 0%, #e8c99a 50%, #c28a45 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px;
    line-height: 1.1;
}
.result-emoji {
    font-size: 3.5rem;
    margin-bottom: 12px;
    display: block;
}
.confidence-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}
.conf-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #c28a45;
    display: inline-block;
}
.conf-text {
    font-size: 13px;
    color: #7a786f;
    font-weight: 400;
}

/* ── Probability bars ── */
.prob-section {
    margin-top: 1.5rem;
    text-align: left;
}
.prob-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a5850;
    margin-bottom: 14px;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-name {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #c8c4bc;
    width: 48px;
    text-transform: capitalize;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e1e24;
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #c28a45, #e8c99a);
    transition: width 0.8s ease;
}
.prob-pct {
    font-size: 12px;
    color: #5a5850;
    width: 38px;
    text-align: right;
    font-weight: 500;
}

/* ── Status/info msgs ── */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #141418;
    border: 1px solid #222228;
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 12px;
    color: #7a786f;
    margin: 0.5rem 0;
}
.dot-green {
    width: 6px; height: 6px;
    background: #5a9e6f;
    border-radius: 50%;
    display: inline-block;
}

/* ── Model not found ── */
.warning-box {
    background: #1a1510;
    border: 1px solid #3a2e1e;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    font-size: 13.5px;
    color: #c8a86a;
    line-height: 1.65;
}
.warning-box code {
    background: #0c0c0f;
    border: 1px solid #2e2a1a;
    border-radius: 6px;
    padding: 1px 7px;
    font-size: 12px;
    color: #d4a96a;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    font-size: 12px;
    color: #3a3830;
    letter-spacing: 0.05em;
}
.footer span { color: #5a5850; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1e1e24;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Model definition (exact copy from notebook) ─────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pooling  = nn.MaxPool2d(2, 2)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(0.4)
        self.flatten  = nn.Flatten()
        self.linear   = nn.Linear(128 * 16 * 16, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.output   = nn.Linear(256, 3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pooling(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pooling(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.linear(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x

# ── Label mapping (LabelEncoder order: alphabetical) ─────────────────────────
# LabelEncoder fits alphabetically: cat=0, dog=1, wild=2
CLASSES      = ["cat", "dog", "wild"]
CLASS_EMOJI  = {"cat": "🐱", "dog": "🐶", "wild": "🦁"}
CLASS_DESC   = {
    "cat":  "Domesticated feline — calm, curious, and graceful.",
    "dog":  "Man's best friend — loyal, playful, and expressive.",
    "wild": "Wild animal — fierce, untamed, and majestic.",
}

# ── Transform (same as transform_val in notebook) ────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
])

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Net().to(device)
    state = torch.load(path, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m, device

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, device, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], probs

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-tag">Deep Learning · CNN · PyTorch</div>
    <h1 class="hero-title">Animal Face<br><span>Classifier</span></h1>
    <p class="hero-sub">Upload a photo of an animal face and the model will predict whether it's a cat, dog, or wild animal.</p>
</div>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
import os

MODEL_PATH = "animal_classifier.pth"
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        model, device = load_model(MODEL_PATH)
        model_loaded = True
        device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <span class="info-pill"><span class="dot-green"></span> Model loaded — running on {device_label}</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>Failed to load model.</strong><br>
            Error: {e}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ <strong>Model file not found.</strong><br><br>
        Place your trained model file at <code>{MODEL_PATH}</code> in the same directory as <code>app.py</code>.<br><br>
        In Google Colab, save and download your model with:<br>
        <code>torch.save(model.state_dict(), "animal_classifier.pth")</code><br>
        <code>from google.colab import files; files.download("animal_classifier.pth")</code>
    </div>
    """, unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:13px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#5a5850; margin:0 0 12px;">Upload Image</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

st.markdown('</div>', unsafe_allow_html=True)

# ── Preview + predict ─────────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")

    # Show preview
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    if st.button("✦  Classify Animal", use_container_width=True):
        if not model_loaded:
            st.error("Model not loaded. Please add the .pth file first.")
        else:
            with st.spinner("Analyzing..."):
                label, probs = predict(model, device, image)

            emoji      = CLASS_EMOJI[label]
            conf_pct   = f"{probs[CLASSES.index(label)]*100:.1f}%"
            description = CLASS_DESC[label]

            # ── Result card ──
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediction result</div>
                <span class="result-emoji">{emoji}</span>
                <div class="result-animal">{label.upper()}</div>
                <p style="font-size:13.5px; color:#7a786f; margin: 8px 0 0; font-weight:300;">{description}</p>
                <div class="confidence-row">
                    <span class="conf-dot"></span>
                    <span class="conf-text">Confidence: <strong style="color:#c8c4bc">{conf_pct}</strong></span>
                </div>

                <hr class="divider">

                <div class="prob-section">
                    <div class="prob-title">All class probabilities</div>
            """, unsafe_allow_html=True)

            for cls in CLASSES:
                idx  = CLASSES.index(cls)
                pct  = probs[idx] * 100
                bar  = f'<div class="prob-bar-fill" style="width:{pct:.1f}%"></div>'
                is_top = cls == label
                name_style = "color:#e8c99a; font-weight:700;" if is_top else ""
                st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-name" style="{name_style}">{CLASS_EMOJI[cls]} {cls}</div>
                        <div class="prob-bar-bg">{bar}</div>
                        <div class="prob-pct">{pct:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <span>PyTorch + Streamlit</span> &nbsp;·&nbsp; AFHQ Dataset &nbsp;·&nbsp; CNN (3-class)
</div>
""", unsafe_allow_html=True)