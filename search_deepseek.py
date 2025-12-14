import json, torch
from pathlib import Path
from modeling_deepseek_chat import DeepseekV2ForCausalLM
from configuration_deepseek_chat import DeepseekV2Config
from transformers import AutoTokenizer
from searching import LayerSkippingSearching
from safetensors import safe_open

model_name = "/data/pretrained_models/DeepSeek-V2-Lite-Chat"   

print(f"Loading tokenizer/model from {model_name} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = DeepseekV2Config.from_pretrained(model_name)
# Force routing to top-1 expert for draft model
config.num_experts_per_tok = 1
config.topk_method = "greedy"
# Sanitize rope_scaling if present (convert ints to floats, clamp factor>=1).
rope_scaling = getattr(config, "rope_scaling", None)
if isinstance(rope_scaling, dict):
    factor = float(max(1.0, rope_scaling.get("factor", 1.0)))
    beta_fast = float(rope_scaling.get("beta_fast", 1.0))
    beta_slow = float(rope_scaling.get("beta_slow", 1.0))
    rope_scaling = {
        "rope_type": rope_scaling.get("rope_type", rope_scaling.get("type", None)),
        "type": rope_scaling.get("type", rope_scaling.get("rope_type", None)),
        "factor": factor,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }
    config.rope_scaling = rope_scaling
    print(f"Using sanitized rope_scaling: {config.rope_scaling}", flush=True)

# Ensure index exists for sharded safetensors
def ensure_index(model_path: Path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return
    shards = sorted(model_path.glob("model-*.safetensors"))
    if not shards:
        return
    weight_map = {}
    for shard in shards:
        with safe_open(shard, framework="np") as f:
            for k in f.keys():
                weight_map[k] = shard.name
    meta = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    index_path.write_text(json.dumps(meta, indent=2))
    print(f"Generated missing index at {index_path}", flush=True)


ensure_index(Path(model_name))

model = DeepseekV2ForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    low_cpu_mem_usage=True,
)
model.eval()
print("Model loaded.", flush=True)

prompts = []
data_path = Path("question.jsonl")
print(f"Loading local prompts from {data_path} ...", flush=True)
if not data_path.exists():
    raise FileNotFoundError(f"{data_path} not found")
with data_path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        turns = obj.get("turns") or []
        if not turns:
            continue
        # join multi-turn into one prompt
        prompt = "\n".join(turns)
        prompts.append(prompt)
# 取前若干条以控制搜索开销
prompts = prompts[:16]
print(f"Total prompts: {len(prompts)}", flush=True)

layer_searching = LayerSkippingSearching(
    model, tokenizer, prompts,
    evaluate_config={"generate_fn": "essg", "max_new_tokens": 32}
)

# layer_searching.probe([], [])

print("Starting search ...", flush=True)
# 正式搜索（n_iter 可先小一点试跑）
layer_searching.search(n_iter=80)
attn_skip, mlp_skip = layer_searching.get_solution()
print("attn:", attn_skip)
print("mlp :", mlp_skip)

# 写回 skip_layers.json
try:
    data = json.load(open("skip_layers.json"))
except FileNotFoundError:
    data = {}
data["deepseek-v2-lite-chat"] = {"attention": attn_skip, "mlp": mlp_skip}
json.dump(data, open("skip_layers.json","w"), indent=2)
print("Saved to skip_layers.json", flush=True)
