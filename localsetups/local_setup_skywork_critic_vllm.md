# Local Setup: Skywork Critic Llama 3.1 8B via vLLM

This file documents the local setup used to run the Skywork Critic model through a vLLM OpenAI-compatible server.

## 1. Start Docker Desktop

Make sure Docker Desktop is running and GPU support is available.

## 2. Start the vLLM server

Run this from WSL/Linux:

```bash
docker run --runtime nvidia --gpus all --rm -it \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Skywork/Skywork-Critic-Llama-3.1-8B \
  --max-model-len 4096 \
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
python scripts/runskyworkcritic_pilot.py   # 10-pair subset
python scripts/runskyworkcritic_full.py    # full dataset
```

The scripts route through `http://localhost:8000/v1` internally — no env vars needed.

## Notes
- Ran on an RTX 3090 Ti; might need to adjust model-len and gpu utilization for different hardware capabilities
- Include the hugging face token setup step before running the model: export HF_TOKEN=hf_your_token_here
- Keep the vLLM server running while the script runs.
- If a run is interrupted, just rerun the script. JudgeBench will resume from where it left off.
