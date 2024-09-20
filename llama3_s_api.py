import os
from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Set the CUDA_VISIBLE_DEVICES environment variable to specify which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 

app = FastAPI()
model_name = "jan-hq/llama3-s-instruct-v0.3-checkpoint-7000"
# Initialize the LLM with tensor parallel degree
llm = LLM(model="/path/to/llama-3.1", tensor_parallel_size=4)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 1024)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling_params)

    return JSONResponse(content={"generated_text": outputs[0].outputs[0].text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)