# clip encode anh + text
"""
embedder.py
Mã hóa ảnh và text thành vector bằng CLIP.
Model được load 1 lần duy nhất khi import module này.
"""

import torch
import open_clip
from PIL import Image

# ── Cấu hình model ──────────────────────────────────────────────────────────
# ViT-B-32: nhẹ, nhanh, đủ tốt cho project nhỏ
# Thay bằng "ViT-L-14" nếu muốn chính xác hơn (cần GPU mạnh hơn)
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Dùng GPU nếu có, ngược lại dùng CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[embedder] Đang load CLIP {MODEL_NAME} trên {DEVICE}...")
_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
_model = _model.to(DEVICE)
_model.eval()
print(f"[embedder] Load xong!")


# ── Public API ───────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> list[float]:
    """
    Đọc file ảnh và trả về vector 512 chiều (đã normalize).
    """
    image = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_image(image)
        # Normalize về unit vector để dùng dot product = cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()


def encode_text(text: str) -> list[float]:
    """
    Encode câu text và trả về vector 512 chiều (đã normalize).
    Nên dùng tiếng Anh để kết quả tốt nhất.
    Nếu query tiếng Việt, dùng translate_and_encode() bên dưới.
    """
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()


def translate_and_encode(text_vi: str) -> list[float]:
    """
    Dịch text tiếng Việt → tiếng Anh rồi encode.
    Fallback về encode_text(text_vi) nếu dịch lỗi.
    """
    try:
        from deep_translator import GoogleTranslator
        text_en = GoogleTranslator(source="vi", target="en").translate(text_vi)
        print(f"  Dịch: '{text_vi}' → '{text_en}'")
        return encode_text(text_en)
    except Exception as e:
        print(f"  [!] Dịch lỗi ({e}), dùng text gốc")
        return encode_text(text_vi)