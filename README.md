<div align="center">
<h1>
    <img src="assets/ssd_logo.png" height="150px" align="top"/>  
    <br>  
    Self-Speculative Decoding
</h1>
</div>
Code associated with the paper:


## Usage

1. Configure the environment according to `ssd.yml` (PyTorch, transformers, bayesian-optimization, safetensors, etc.).
2. (LLaMA family) Use `search.ipynb` to search skipped layers and `evaluate_*` notebooks for evaluation.
3. (DeepSeek V2 Lite Chat) Use the provided script to search skipped layers on a local prompt set:

   - Prepare `question.jsonl` in repo root (already included; uses `turns` field as prompts).
   - Load model weights under `model_name` (default `/data/pretrained_models/DeepSeek-V2-Lite-Chat`).
   - Run:
     ```bash
     # optional: limit GPUs
     export CUDA_VISIBLE_DEVICES=0,1,2,3
     python search_deepseek.py
     ```
   - The script will:
     - load tokenizer/model with `device_map="auto"` (multi-GPU/CPU as needed);
     - build prompts from `question.jsonl`;
     - run Bayesian optimization over skip-layer combinations (default `n_iter=200`, generate_fn `essg`, `max_new_tokens=32`);
     - print per-eval throughput and save the best skip sets to `skip_layers.json` under key `deepseek-v2-lite-chat`.

Notes:
- If you want faster trials, reduce `n_iter`, truncate prompts, or lower `max_new_tokens`.
- Current `essg` run disables KV cache for stability; throughput numbers are for this setting.
