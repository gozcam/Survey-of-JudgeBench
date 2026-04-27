# Local Setup: Skywork Reward Llama 3.1 8B via CUDA

This file documents the local setup used to run the Skywork Reward model directly with CUDA.

## 1. Activate the virtual environment

```bash
source .venv/bin/activate     # WSL/Linux
```

## 2. Install CUDA/local-model requirements

```bash
pip install -r requirements-cuda.txt
```

If `flash-attn` fails to install, (which happened to me) install the rest first then retry:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install datasets transformers sentencepiece accelerate huggingface_hub
pip install "httpx<0.28"
```

If flash-attn fails to install, do not block the experiment on it. In the reward-model loader, change:

attn_implementation="flash_attention_2" 

to:

attn_implementation="sdpa"

or remove the attn_implementation line entirely

## 3. Verify the GPU is visible to PyTorch

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output: `True` followed by your GPU name.

## 4. Run the JudgeBench script

```bash
python scripts/runskyworkreward_pilot.py   # 10-pair subset
python scripts/runskyworkreward_full.py    # full dataset
```

## Notes
- Ran on an RTX 3090 Ti
- Include the hugging face token setup step before running the model: export HF_TOKEN=hf_your_token_here
- The reward model scores response A and B independently and picks the higher-scored one — this is different from the text-verdict prompted judges.
- Because scoring is single-pass, the inconsistency metric does not apply.
- The model will be downloaded from HuggingFace on the first run.
