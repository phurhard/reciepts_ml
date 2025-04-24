# Receipt ML

A machine learning system that differentiates between AI-generated receipts and authentic receipts using a hybrid deep learning approach.

## Overview

This project uses a combination of computer vision and natural language processing techniques to analyze receipt images and determine whether they are genuine or AI-generated. The system employs a hybrid model architecture that processes both visual features and extracted text to make accurate classifications.

## Key Features

- **Hybrid Model Architecture**: Combines image features (EfficientNet-B0) and text features (BERT) for comprehensive analysis
- **Advanced OCR**: Uses TrOCR (Transformer OCR) to extract text from receipt images
- **Multi-format Support**: Handles both image files and PDF documents
- **REST API**: Provides a FastAPI endpoint for easy integration
- **Detailed Output**: Returns classification result, confidence scores, and extracted text

## Technology Stack

- **Deep Learning**: PyTorch, Transformers
- **Computer Vision**: EfficientNet, TorchVision
- **NLP**: BERT, Hugging Face Transformers
- **OCR**: TrOCR, pytesseract
- **API**: FastAPI
- **PDF Processing**: pdf2image

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/phurhard/receipt_ml.git
   cd receipt_ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the model files in the `model/` directory.

## Usage

### Starting the API Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

### Making Predictions

Send a POST request to the `/predict` endpoint with a receipt image:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/receipt.jpg"
```

### API Response

```json
{
  "prediction": "real",
  "probabilities": {
    "ai_generated": 0.12,
    "real": 0.88
  },
  "extracted_text": "GROCERY STORE\nMilk $3.99\nBread $2.49\n..."
}
```

## Dataset

The project uses a dataset of real and AI-generated receipts organized in the following structure:
- `receipt_dataset/real/` - Contains authentic receipt images
- `receipt_dataset/ai_generated/` - Contains AI-generated receipt images

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
