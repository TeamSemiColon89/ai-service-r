# AI Service (Rust)

This is a simple AI-powered web service for detecting SQL injection (SQLi) in input text using a DistilBERT ONNX model and a HuggingFace tokenizer.

## Features
- REST API endpoint for SQLi detection
- Fast inference using ONNX Runtime
- Lightweight Actix-web server
- Docker support for easy deployment

## How to Run

### 1. Run Locally (Requires Rust)
```sh
cargo run --release
```
The service will be available at: http://localhost:5000/decision

### 2. Run with Docker
Build the image:
```sh
docker build -t ai-service-rust:latest .
```
Run the container:
```sh
docker run -p 5000:5000 ai-service-rust:latest
```

## Usage
Send a POST request to `/decision` with your input as the raw body (plain text):

```sh
curl -X POST http://localhost:5000/decision -H "Content-Type: text/plain" -d 'id=1 OR 1=1'
```

**Response:**
```json
{
	"input": "id=1 OR 1=1",
	"prediction": "SQLi",
	"logits": [ ... ]
}
```

## Notes
- The ONNX model and tokenizer files must be present in the `models/` and `tokenizer/` folders.
- The service listens on port 5000 by default.
# ai-service-r
