model_name="jan-hq/llama3-s-instruct-v0.3-checkpoint-7000"
api_key="jan0974980045"
port=3347
vllm serve $model_name --dtype bfloat16 --port 3347