import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from contextlib import asynccontextmanager
import uvicorn

# Global variables to store the model
llm = None
tokenizer = None
params = None
happy_token = None
sad_token = None
neutral_token = None


def softmax(x, temp=1.0):
    """Apply softmax function with temperature scaling"""
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    x = np.array(x) / temp
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def fmt(sentence):
    """Format sentence for emotion classification"""
    sentence = sentence.strip()
    # Create a structured message for classification
    messages = [
        {"role": "user", "content": f"/no_think Classify this sentence into: Happy, Sad, Neutral.\n{sentence}"},
        {"role": "assistant", "content": ""}
    ]

    # Apply the model's chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback for tokenizers without chat template
        formatted_text = "/no_think Classify this sentence into: Happy, Sad, Neutral.\n" + sentence + "\nAssistant: "

    return formatted_text


async def load_model():
    """Load the emotion classification model"""
    global llm, tokenizer, params, happy_token, sad_token, neutral_token

    model_path = "hoanganhpham/text_emotion_0.6B"

    print("Loading model...")
    llm = LLM(
        model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_seqs=128,
        dtype=torch.float16,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=True,
        max_logprobs=15000
    )

    tokenizer = llm.get_tokenizer()

    # Get token IDs for emotion classes
    happy_token = tokenizer("Happy", return_tensors="pt").input_ids.item()
    sad_token = tokenizer("Sad", return_tensors="pt").input_ids.item()
    neutral_token = tokenizer("Neutral", return_tensors="pt").input_ids.item()

    # Set up sampling parameters
    params = SamplingParams(
        n=1,
        top_k=1,
        temperature=0,
        seed=777,
        skip_special_tokens=False,
        max_tokens=1,
        logprobs=15000
    )

    print("Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    pass

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Classification API",
    description="API for classifying text emotions into Happy, Sad, or Neutral",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class TextInput(BaseModel):
    text: str


class EmotionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    probabilities: dict
    processing_time: float


# Emotion mapping
EMOTION_MAPPING = {
    0: "Happy",
    1: "Sad", 
    2: "Neutral"
}


@app.post("/classify", response_model=EmotionResponse)
async def classify_emotion(input_data: TextInput):
    """
    Classify the emotion of input text

    Returns:
    - emotion: Predicted emotion (Happy, Sad, Neutral)
    - confidence: Confidence score for the prediction
    - probabilities: Probability distribution over all emotions
    - processing_time: Time taken for processing in seconds
    """
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        start_time = time.time()

        # Format the input text
        formatted_prompt = fmt(input_data.text)

        # Generate prediction
        outputs = llm.generate(formatted_prompt, params)

        # Extract logprobs for each emotion
        output = outputs[0]
        logprobs = [
            output.outputs[0].logprobs[0].get(happy_token),
            output.outputs[0].logprobs[0].get(sad_token), 
            output.outputs[0].logprobs[0].get(neutral_token)
        ]

        # Convert to actual logprob values
        logprobs = [x.logprob if x is not None else -np.inf for x in logprobs]

        # Apply softmax to get probabilities
        probabilities = softmax(logprobs)

        # Add slight bias to neutral (as in original code)
        probabilities[2] += 0.001

        # Get prediction
        pred_idx = np.argmax(probabilities)
        predicted_emotion = EMOTION_MAPPING[pred_idx]
        confidence = float(probabilities[pred_idx])

        processing_time = time.time() - start_time

        # Create probability dictionary
        prob_dict = {
            "Happy": float(probabilities[0]),
            "Sad": float(probabilities[1]),
            "Neutral": float(probabilities[2])
        }

        return EmotionResponse(
            text=input_data.text,
            emotion=predicted_emotion,
            confidence=confidence,
            probabilities=prob_dict,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "timestamp": time.time()
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Emotion Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/classify": "POST - Classify emotion of text",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False for production
        workers=1  # Single worker due to GPU model
    )
