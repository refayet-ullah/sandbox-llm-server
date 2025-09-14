from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI(title="LLM Inference Server")

# Configure CORS to allow requests from your Node.js server later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"], # URLs of your Next.js and Node.js servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM model
model_path = "./Meta-Llama-3-8B-Instruct.Q4_0.gguf"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}. Please download it.")

print("Loading LLM model... This may take a while.")
llm = Llama(
    model_path=model_path,
    n_ctx=7168,        # Context window. Lower if you have less RAM.
    n_threads=8,       # Number of CPU threads to use
    n_gpu_layers=0,    # Number of layers to offload to GPU (if you have one). Set to 0 for CPU-only.
    verbose=False
)
print("Model loaded successfully!")

# Define the request body model
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2
    stop: list = ["<|eot_id|>"] # Llama 3's stop token

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        # Generate response
        output = llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            echo=False
        )
        
        # Extract the text from the response
        completion_text = output['choices'][0]['text'].strip()
        return {"response": completion_text, "usage": output.get('usage', {})}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": model_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)