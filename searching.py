import torch
from decoding import infer
import json

from bayes_opt import BayesianOptimization


class LayerSkippingSearching:
    def __init__(
        self,
        model,
        tokenizer,
        evaluate_prompts,
        evaluate_config={"generate_fn": "essg", "max_new_tokens": 32},
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.evaluate_prompts = evaluate_prompts
        self.evaluate_config = evaluate_config
        self._eval_count = 0
        # Prefer skipping later layers; enforce attn/mlp skip ids to match.
        # Only consider roughly the last 1/3 layers.
        self.min_skip_layer = max(int(self.config.num_hidden_layers * 0.5), self.config.num_hidden_layers - 10)
        self.candidate_layers = list(range(self.min_skip_layer, self.config.num_hidden_layers))

        # One variable per candidate layer (attn/mlp share the same id)
        self.pbounds = {f"x{i}": (0, 1) for i in range(len(self.candidate_layers))}

        self.optimizer = BayesianOptimization(
            f=self._black_box_evaluate_function, pbounds=self.pbounds, random_state=1, verbose=1
        )

        self.optimizer.set_gp_params(alpha=1e-2)

    def _black_box_evaluate_function(self, **kargs):
        self._eval_count += 1
        attn_skip_layers = []
        for idx, layer_id in enumerate(self.candidate_layers):
            if kargs[f"x{idx}"] > 0.5:
                attn_skip_layers.append(layer_id)
        # Enforce mlp skip ids to be identical to attn skip ids
        mlp_skip_layers = attn_skip_layers.copy()

        self.model.set_skip_layers(
            attn_skip_layer_id_set=attn_skip_layers,
            mlp_skip_layer_id_set=mlp_skip_layers,
        )

        total_time = 0
        total_tokens = 0
        matchness_list = []

        for prompt in self.evaluate_prompts:
            ret = infer(self.model, self.tokenizer, prompt, **self.evaluate_config)
            total_time += ret["time"]
            total_tokens += self.evaluate_config.get("max_new_tokens", 10)
            if "matchness" in ret:
                matchness_list.append(ret["matchness"])

        avg_matchness = sum(matchness_list) / len(matchness_list) if matchness_list else 1.0

        tokens_per_s = total_tokens / total_time
        # Combine速度与质量：提高质量权重，并对跳层数加轻微惩罚
        skip_penalty = 0.01 * (len(attn_skip_layers) + len(mlp_skip_layers))
        if avg_matchness < 0.5:
            score = tokens_per_s * (avg_matchness ** 2) - skip_penalty - 1.0  # 强惩罚低接受率
        else:
            score = tokens_per_s * (avg_matchness ** 2) - skip_penalty

        print(
            f"[Eval {self._eval_count}]",
            f"{tokens_per_s:.3f} tokens/s",
            f"matchness={avg_matchness:.3f}",
            f"score={score:.3f}",
            "Skipped attn:",
            len(attn_skip_layers),
            "Skipped mlp:",
            len(mlp_skip_layers),
            f"attn_ids={attn_skip_layers}",
        )

        return score

    def probe(self, attn_skip_layers, mlp_skip_layers):
        """
        Add some good points to accelerate searching
        """

        params = {f"x{i}": 0.0 for i in range(len(self.candidate_layers))}
        for lid in attn_skip_layers:
            if lid in self.candidate_layers:
                idx = self.candidate_layers.index(lid)
                params[f"x{idx}"] = 1.0
        self.optimizer.probe(params=params, lazy=True)

    def search(self, n_iter=1000):
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        return self.get_solution()

    def get_solution(self):

        skip_attn_layers = []
        for idx, layer_id in enumerate(self.candidate_layers):
            if self.optimizer.max["params"][f"x{idx}"] > 0.5:
                skip_attn_layers.append(layer_id)

        skip_mlp_layers = skip_attn_layers.copy()

        return skip_attn_layers, skip_mlp_layers
