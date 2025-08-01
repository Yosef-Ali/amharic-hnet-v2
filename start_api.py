#!/usr/bin/env python3
"""
Simple API server for Amharic H-Net using the retrained model
"""
import sys
import os
sys.path.append('src')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from typing import Optional
import json

from models.hnet_amharic import AmharicHNet

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.8
    top_k: int = 30

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    success: bool
    amharic_ratio: float

# Initialize FastAPI app
app = FastAPI(
    title="Amharic H-Net API",
    description="Production API for Amharic text generation using H-Net architecture",
    version="1.1.0"
)

# Global model variable
model = None

def load_model():
    """Load the morpheme-supervised Amharic H-Net model"""
    global model
    
    # Try to load the latest sentence continuation model first
    sentence_model_path = "outputs/sentence_continuation_model.pt"
    morpheme_model_path = "outputs/morpheme_supervised_model.pt"
    fallback_path = "outputs/compact/final_model.pt"
    
    if os.path.exists(sentence_model_path):
        selected_path = sentence_model_path
        model_type = "Sentence-Continuation"
    elif os.path.exists(morpheme_model_path):
        selected_path = morpheme_model_path
        model_type = "Morpheme-Supervised"
    elif os.path.exists(fallback_path):
        selected_path = fallback_path
        model_type = "Standard"
    else:
        raise FileNotFoundError(f"No model found at {sentence_model_path}, {morpheme_model_path}, or {fallback_path}")
    
    # Create model with correct architecture
    model = AmharicHNet(
        d_model=256,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_main_layers=4,
        n_heads=4,
        vocab_size=256
    )
    
    # Load checkpoint
    checkpoint = torch.load(selected_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ {model_type} Model loaded successfully from {selected_path}")
    if 'final_loss' in checkpoint:
        print(f"📊 Training loss: {checkpoint['final_loss']:.4f}")
    if 'boundary_accuracy' in checkpoint:
        print(f"🎯 Boundary accuracy: {checkpoint['boundary_accuracy']:.1%}")
    if 'training_type' in checkpoint:
        print(f"🧠 Training type: {checkpoint['training_type']}")
    if 'examples_trained' in checkpoint:
        print(f"📚 Examples trained: {checkpoint['examples_trained']}")
    
    return model

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Amharic H-Net API - Sentence Continuation",
        "version": "2.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "capabilities": ["morpheme_analysis", "sentence_continuation", "meaningful_amharic_generation"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "amharic-hnet-api"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate Amharic text using the H-Net model"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert prompt to bytes
        prompt_bytes = request.prompt.encode('utf-8')
        input_ids = torch.tensor([[b for b in prompt_bytes]], dtype=torch.long)
        
        # Generate text
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=min(request.max_length, 50),  # Smaller limit to avoid repetition
                temperature=request.temperature,
                top_k=request.top_k
            )
        
        # Convert back to text with better handling
        generated_bytes = generated[0].cpu().numpy()
        
        # Filter out null bytes and control characters
        filtered_bytes = []
        for byte_val in generated_bytes:
            if 32 <= byte_val <= 126 or byte_val >= 128:  # Printable ASCII + non-ASCII
                filtered_bytes.append(byte_val)
            elif byte_val in [32, 10, 13]:  # Space, newline, carriage return
                filtered_bytes.append(byte_val)
        
        # Convert to text
        try:
            generated_text = bytes(filtered_bytes).decode('utf-8', errors='ignore')
        except:
            # Fallback: use original prompt if decoding fails completely
            generated_text = request.prompt + "..."
        
        # Remove excessive repetition
        if len(generated_text) > len(request.prompt) * 3:
            generated_text = generated_text[:len(request.prompt) * 3]
        
        # Ensure we have at least the original prompt
        if len(generated_text) < len(request.prompt):
            generated_text = request.prompt
        
        # Calculate Amharic ratio
        amharic_chars = sum(1 for c in generated_text if 0x1200 <= ord(c) <= 0x137F)
        total_chars = len(generated_text)
        amharic_ratio = amharic_chars / total_chars if total_chars > 0 else 0.0
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            success=True,
            amharic_ratio=amharic_ratio
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024
    
    return {
        "architecture": "AmharicHNet",
        "parameters": total_params,
        "size_mb": round(model_size_mb, 1),
        "d_model": model.d_model,
        "compression_ratio": model.compression_ratio,
        "morpheme_patterns": {
            "verb_prefixes": model.chunker.verb_prefixes,
            "verb_suffixes": model.chunker.verb_suffixes[:5],  # Show first 5
            "noun_prefixes": model.chunker.noun_prefixes[:5],
            "noun_suffixes": model.chunker.noun_suffixes[:5],
            "total_patterns": len(model.chunker.verb_prefixes + model.chunker.verb_suffixes + 
                                model.chunker.noun_prefixes + model.chunker.noun_suffixes)
        }
    }

@app.post("/analyze-morphemes")
async def analyze_morphemes(text: str):
    """Analyze morpheme boundaries in Amharic text"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert text to bytes
        text_bytes = text.encode('utf-8')
        input_ids = torch.tensor([[b for b in text_bytes]], dtype=torch.long)
        
        # Get embeddings and analyze boundaries
        with torch.no_grad():
            embeddings = model.byte_embedding(input_ids)
            boundary_probs = model.chunker(embeddings)
        
        # Extract boundary analysis
        probs = boundary_probs[0].tolist()
        boundaries = [i for i, p in enumerate(probs) if p > 0.5]
        
        # Calculate morpheme count
        morpheme_count = len(boundaries)
        
        # Check for known morphological patterns
        patterns_found = []
        for prefix in model.chunker.verb_prefixes + model.chunker.noun_prefixes:
            if text.startswith(prefix):
                patterns_found.append(f"prefix:{prefix}")
        
        for suffix in model.chunker.verb_suffixes + model.chunker.noun_suffixes:
            if text.endswith(suffix):
                patterns_found.append(f"suffix:{suffix}")
        
        return {
            "text": text,
            "byte_length": len(text_bytes),
            "boundary_probabilities": [round(p, 3) for p in probs],
            "boundary_positions": boundaries,
            "predicted_morphemes": morpheme_count,
            "patterns_detected": patterns_found,
            "analysis": {
                "avg_boundary_confidence": round(sum(probs) / len(probs), 3),
                "max_boundary_confidence": round(max(probs), 3),
                "morphological_complexity": "high" if morpheme_count > 3 else "medium" if morpheme_count > 1 else "low"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Morpheme analysis failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Amharic H-Net API Server")
    print("=" * 40)
    
    # Load model before starting server
    try:
        load_model()
        print("🌟 Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)
    
    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=8000)