import argparse
import json
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from generalization.data.dataset import AdditionDataset
from generalization.data.tokenizer import AdditionTokenizer
from generalization.models.models import AdditionTransformer


DEFAULT_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "=": 11,
    "<eos>": 12,
    "<pad>": 13,
}


@dataclass
class LensConfig:
    batch_size: int
    device: str
    checkpoint_path: str
    save_dir: str
    model_config: dict
    tokenizer_config: dict
    dataset_config: dict
    include_eos: bool = True
    strict_example_index: int = 0


class AdditionLogitLens:
    def __init__(self, config: LensConfig):
        self.config = config
        self.device = config.device

        self.model = AdditionTransformer(**config.model_config).to(self.device)
        state_dict = torch.load(config.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.tokenizer = AdditionTokenizer(**config.tokenizer_config)
        self.dataset = AdditionDataset(**config.dataset_config)

    def _project_hidden_states(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        projected = []
        for state in hidden_states:
            projected.append(self.model.output(self.model.norm(state)))

        return torch.stack(projected, dim=0)

    def _decode_token(self, token_id: int) -> str:
        token = self.tokenizer.inv_vocab[token_id]
        if token == "<eos>":
            return "eos"
        if token == "<pad>":
            return "pad"
        return token

    def run(self) -> dict:
        os.makedirs(self.config.save_dir, exist_ok=True)

        eval_samples = self.dataset.data
        full_sequences = [
            f"{sample['question']}{sample['ground_truth']}"
            for sample in eval_samples
        ]

        encoded = self.tokenizer.encode(full_sequences)
        input_tokens = encoded[:, :-1].to(self.device)
        label_tokens = encoded[:, 1:].to(self.device)
        attention_mask = input_tokens != self.tokenizer.padding_id

        question_lengths = torch.tensor(
            [len(sample["question"]) for sample in eval_samples],
            device=self.device,
        )
        answer_lengths = torch.tensor(
            [len(sample["ground_truth"]) for sample in eval_samples],
            device=self.device,
        )

        num_layers = len(self.model.blocks) + 1
        max_answer_steps = int(answer_lengths.max().item()) + int(self.config.include_eos)

        log_prob_sum = torch.zeros(num_layers, max_answer_steps, dtype=torch.float64)
        accuracy_sum = torch.zeros_like(log_prob_sum)
        count_sum = torch.zeros_like(log_prob_sum)

        sample_details = []
        strict_example = None

        with torch.no_grad():
            for start in range(0, len(eval_samples), self.config.batch_size):
                end = min(start + self.config.batch_size, len(eval_samples))

                batch_inputs = input_tokens[start:end]
                batch_labels = label_tokens[start:end]
                batch_mask = attention_mask[start:end]
                batch_question_lengths = question_lengths[start:end]
                batch_answer_lengths = answer_lengths[start:end]

                _, hidden_states = self.model(
                    batch_inputs,
                    batch_mask,
                    return_hidden_states=True,
                )
                layer_logits = self._project_hidden_states(hidden_states)
                layer_log_probs = F.log_softmax(layer_logits, dim=-1)
                layer_predictions = layer_logits.argmax(dim=-1)

                batch_size, seq_len = batch_labels.shape
                positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
                answer_start_positions = batch_question_lengths.unsqueeze(1) - 1
                answer_steps = positions - answer_start_positions

                valid_answer_mask = answer_steps >= 0
                if self.config.include_eos:
                    valid_answer_mask &= answer_steps <= batch_answer_lengths.unsqueeze(1)
                else:
                    valid_answer_mask &= answer_steps < batch_answer_lengths.unsqueeze(1)

                non_pad_mask = batch_labels != self.tokenizer.padding_id
                valid_answer_mask &= non_pad_mask

                expanded_labels = batch_labels.unsqueeze(0).unsqueeze(-1).expand(num_layers, -1, -1, 1)
                gathered_log_probs = layer_log_probs.gather(dim=-1, index=expanded_labels).squeeze(-1)
                gathered_logits = layer_logits.gather(dim=-1, index=expanded_labels).squeeze(-1)
                correct = layer_predictions.eq(batch_labels.unsqueeze(0))

                for step_idx in range(max_answer_steps):
                    step_mask = valid_answer_mask & (answer_steps == step_idx)
                    if not step_mask.any():
                        continue

                    expanded_step_mask = step_mask.unsqueeze(0).expand(num_layers, -1, -1)
                    log_prob_sum[:, step_idx] += gathered_log_probs.masked_select(expanded_step_mask).reshape(num_layers, -1).sum(dim=1).cpu()
                    accuracy_sum[:, step_idx] += correct.masked_select(expanded_step_mask).reshape(num_layers, -1).float().sum(dim=1).cpu()
                    count_sum[:, step_idx] += step_mask.sum().item()

                if not sample_details:
                    sample_logits = layer_logits[:, 0].cpu()
                    sample_steps = valid_answer_mask[0].nonzero(as_tuple=False).flatten().tolist()
                    sample_details.append({
                        "question": eval_samples[start]["question"],
                        "ground_truth": eval_samples[start]["ground_truth"],
                        "target_tokens": [
                            self._decode_token(int(batch_labels[0, pos].item()))
                            for pos in sample_steps
                        ],
                        "top_predictions": [
                            [
                                self._decode_token(int(sample_logits[layer_idx, pos].argmax().item()))
                                for pos in sample_steps
                            ]
                            for layer_idx in range(num_layers)
                        ],
                    })

                strict_index = self.config.strict_example_index
                if strict_example is None and start <= strict_index < end:
                    batch_idx = strict_index - start
                    sample_step_positions = valid_answer_mask[batch_idx].nonzero(as_tuple=False).flatten()
                    strict_log_probs = gathered_log_probs[:, batch_idx, sample_step_positions].cpu()
                    strict_logits = gathered_logits[:, batch_idx, sample_step_positions].cpu()
                    strict_correct = correct[:, batch_idx, sample_step_positions].float().cpu()
                    strict_target_token_ids = [
                        int(batch_labels[batch_idx, pos].item())
                        for pos in sample_step_positions.tolist()
                    ]

                    strict_example = {
                        "dataset_index": strict_index,
                        "question": eval_samples[strict_index]["question"],
                        "ground_truth": eval_samples[strict_index]["ground_truth"],
                        "step_positions": sample_step_positions.cpu().tolist(),
                        "target_token_ids": strict_target_token_ids,
                        "target_tokens": [
                            self._decode_token(token_id) for token_id in strict_target_token_ids
                        ],
                        "log_prob": strict_log_probs.tolist(),
                        "logit": strict_logits.tolist(),
                        "is_top1": strict_correct.tolist(),
                        "top_predictions": [
                            [
                                self._decode_token(
                                    int(layer_predictions[layer_idx, batch_idx, pos].item())
                                )
                                for pos in sample_step_positions.tolist()
                            ]
                            for layer_idx in range(num_layers)
                        ],
                    }

        mean_log_prob = log_prob_sum / count_sum.clamp_min(1.0)
        mean_accuracy = accuracy_sum / count_sum.clamp_min(1.0)

        step_labels = [f"ans_{idx + 1}" for idx in range(max_answer_steps - int(self.config.include_eos))]
        if self.config.include_eos:
            step_labels.append("eos")

        results = {
            "config": asdict(self.config),
            "layer_labels": [f"embed" if idx == 0 else f"layer_{idx}" for idx in range(num_layers)],
            "step_labels": step_labels,
            "mean_log_prob": mean_log_prob.tolist(),
            "mean_accuracy": mean_accuracy.tolist(),
            "counts": count_sum.tolist(),
            "sample_details": sample_details,
            "strict_example": strict_example,
        }

        torch.save(
            {
                "mean_log_prob": mean_log_prob,
                "mean_accuracy": mean_accuracy,
                "counts": count_sum,
                "results": results,
            },
            os.path.join(self.config.save_dir, "logit_lens.pt"),
        )

        with open(os.path.join(self.config.save_dir, "logit_lens.json"), "w") as file:
            json.dump(results, file, indent=2)

        self._save_heatmap(
            mean_log_prob,
            results["layer_labels"],
            step_labels,
            "Mean log p(ground truth token)",
            os.path.join(self.config.save_dir, "logit_lens_log_prob.png"),
        )
        self._save_heatmap(
            mean_accuracy,
            results["layer_labels"],
            step_labels,
            "Ground-truth top-1 accuracy",
            os.path.join(self.config.save_dir, "logit_lens_accuracy.png"),
            vmin=0.0,
            vmax=1.0,
        )

        if strict_example is not None:
            strict_step_labels = strict_example["target_tokens"]
            self._save_heatmap(
                torch.tensor(strict_example["log_prob"], dtype=torch.float32),
                results["layer_labels"],
                strict_step_labels,
                f"Strict example log p(gt token) #{strict_example['dataset_index']}",
                os.path.join(self.config.save_dir, "logit_lens_strict_log_prob.png"),
            )
            self._save_heatmap(
                torch.tensor(strict_example["logit"], dtype=torch.float32),
                results["layer_labels"],
                strict_step_labels,
                f"Strict example gt-token logit #{strict_example['dataset_index']}",
                os.path.join(self.config.save_dir, "logit_lens_strict_logit.png"),
            )
            self._save_heatmap(
                torch.tensor(strict_example["is_top1"], dtype=torch.float32),
                results["layer_labels"],
                strict_step_labels,
                f"Strict example top-1 correctness #{strict_example['dataset_index']}",
                os.path.join(self.config.save_dir, "logit_lens_strict_top1.png"),
                vmin=0.0,
                vmax=1.0,
            )

        return results

    def _save_heatmap(
        self,
        matrix: torch.Tensor,
        layer_labels: list[str],
        step_labels: list[str],
        title: str,
        save_path: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        figure, axis = plt.subplots(figsize=(max(6, len(step_labels) * 1.1), max(5, len(layer_labels) * 0.55)))
        image = axis.imshow(matrix.numpy(), aspect="auto", cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(step_labels)))
        axis.set_xticklabels(step_labels)
        axis.set_yticks(range(len(layer_labels)))
        axis.set_yticklabels(layer_labels)
        axis.set_xlabel("Prediction step")
        axis.set_ylabel("Residual stream snapshot")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(save_path, dpi=180)
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a logit-lens style view over the addition model.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--min-digits", type=int, default=4)
    parser.add_argument("--max-digits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=728)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--strict-example-index", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()

    config = LensConfig(
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir,
        include_eos=args.include_eos,
        strict_example_index=args.strict_example_index,
        model_config={
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "emb_dim": args.emb_dim,
            "ffn_mult": args.ffn_mult,
            "vocab_size": len(DEFAULT_VOCAB),
            "max_seq_len": args.max_seq_len,
        },
        tokenizer_config={
            "vocab": DEFAULT_VOCAB,
            "eos_id": DEFAULT_VOCAB["<eos>"],
            "padding_id": DEFAULT_VOCAB["<pad>"],
        },
        dataset_config={
            "num_samples": args.num_samples,
            "num_digits": [args.min_digits, args.max_digits],
            "seed": args.seed,
            "mode": "eval",
        },
    )

    lens = AdditionLogitLens(config)
    results = lens.run()
    print(json.dumps({
        "save_dir": config.save_dir,
        "num_layers": len(results["layer_labels"]),
        "num_steps": len(results["step_labels"]),
    }, indent=2))


if __name__ == "__main__":
    main()
