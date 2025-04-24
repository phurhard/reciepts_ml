from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
from torchvision import transforms, models
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, BertTokenizer, BertModel

app = FastAPI()

# Device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# ----------------------- Load Models -----------------------

# Image Model (EfficientNet-B0 without classifier)
image_model = models.efficientnet_b0(pretrained=False)
image_model.classifier = nn.Identity()
checkpoint = torch.load("model/best_hybrid_model.pth", map_location=DEVICE)
image_model.load_state_dict(checkpoint['image_model'])
image_model.to(DEVICE).eval()

# Text Model (BERT encoder)
text_model = BertModel.from_pretrained('bert-base-uncased')
text_model.load_state_dict(checkpoint['text_model'])
text_model.to(DEVICE).eval()

# OCR Processor & Model (TrOCR)
ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(DEVICE).eval()

# Text Tokenizer
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Hybrid Classifier Definition
class HybridClassifier(nn.Module):
    def __init__(self, img_feat_dim=1280, txt_feat_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_feat_dim + txt_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_feats, txt_feats):
        x = torch.cat([img_feats, txt_feats], dim=1)
        return self.fc(x)

hybrid_model = HybridClassifier().to(DEVICE)
hybrid_model.load_state_dict(checkpoint['hybrid_model'])
hybrid_model.eval()

# --------------------- Image & Text Transforms ---------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------- Prediction Endpoint -----------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file bytes
    contents = await file.read()

    # Handle PDF: convert first page to image
    if file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
        try:
            pages = convert_from_bytes(contents)
            image = pages[0].convert("RGB")
        except Exception:
            return JSONResponse({"error": "Invalid PDF format"}, status_code=400)
    else:
        # Handle image formats
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse({"error": "Invalid image format"}, status_code=400)

    # Preprocess image and extract features
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_feats = image_model(img_tensor)

    # OCR text extraction with TrOCR
    pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Encode text features
    encoding = text_tokenizer.encode_plus(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        txt_out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = txt_out.pooler_output
        logits = hybrid_model(img_feats, txt_feats)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred_idx = int(torch.argmax(torch.tensor(probs), dim=0).item())

    # Map prediction
    label_map = {0: "ai_generated", 1: "real"}
    return JSONResponse({
        "prediction": label_map[pred_idx],
        "probabilities": {"ai_generated": probs[0], "real": probs[1]},
        "extracted_text": text
    })
