# Local Setup: Llama 3.1 8B Instruct via vLLM

This file documents the local setup used to run the prompted Llama judge through a vLLM OpenAI-compatible server.

## 1. Start Docker Desktop

Make sure Docker Desktop is running and GPU support is available.

## 2. Start the vLLM server

Run this from WSL/Linux:

```bash
docker run --runtime nvidia --gpus all --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.90
```

## 3. Confirm the server is running

In another terminal:

```bash
curl http://localhost:8000/v1/models
```

## 4. Run the JudgeBench script

With the server running, execute from the repo root:

```bash
python scripts/runllama31_8b_pilot.py   # 10-pair subset
python scripts/runllama31_8b_full.py    # full dataset
```

The scripts route through `http://localhost:8000/v1` internally — no env vars needed.

## Notes
- Ran on an RTX 3090 Ti; might need to adjust model-len and gpu utilization for different hardware capabilities
- Include the hugging face token setup step before running the model: export HF_TOKEN=hf_your_token_here
- Keep the vLLM Docker terminal open while the script runs.
- If a run is interrupted, just rerun the script. JudgeBench will resume from where it left off.
