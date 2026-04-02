import torch
import torch.nn.functional as F

from generalization.data.dataloader import AdditionDataloader
from generalization.data.dataset import AdditionDataset
from generalization.models.models import AdditionPredicter, AdditionTransformer
from generalization.data.tokenizer import AdditionTokenizer


class AdditionEval():
    def __init__(self,
                 batch_size: int, model_config: dict, 
                 tokenizer_config: dict, dataset_config: dict,
                 device: str, checkpoint_path: str,
                 save_path: str, save_name: str
    ):
        self.device = device
        self.model = AdditionTransformer(**model_config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.tokenizer = AdditionTokenizer(**tokenizer_config)
        self.dataset = AdditionDataset(**dataset_config)
        self.dataloader = AdditionDataloader(self.tokenizer, self.dataset, batch_size, "eval")
        self.save_path = save_path
        self.save_name = save_name

    def generate(self, input: list[str]) -> list[str]:
        # this is really bad but it's 4am and i'm on
        # a bus, so some ops will not be vectorized

        tokenized_input = self.tokenizer.encode(input).to(self.device)
        tokenized_input = tokenized_input[:, :-1]

        max_len = self.model.pos_emb.num_embeddings
        outputs: list[torch.Tensor | None] = [None] * len(input)
        active_indices = torch.arange(len(input), device=self.device)

        self.model.eval()
        with torch.no_grad():
            while tokenized_input.size(0) > 0 and tokenized_input.size(1) < max_len:
                attention_mask = tokenized_input != self.tokenizer.padding_id
                logits = self.model(tokenized_input, attention_mask)
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                tokenized_input = torch.cat([tokenized_input, next_tokens.unsqueeze(1)], dim=1)

                finished_mask = next_tokens == self.tokenizer.eos_id

                for batch_idx, original_idx in enumerate(active_indices[finished_mask].tolist()):
                    outputs[original_idx] = tokenized_input[finished_mask][batch_idx].detach().cpu()

                alive_mask = ~finished_mask
                tokenized_input = tokenized_input[alive_mask]
                active_indices = active_indices[alive_mask]

        for batch_idx, original_idx in enumerate(active_indices.tolist()):
            outputs[original_idx] = tokenized_input[batch_idx].detach().cpu()

        decoded_outputs = []
        for output in outputs:
            cleaned = []
            for token in output.tolist():
                if token == self.tokenizer.padding_id:
                    continue
                if token == self.tokenizer.eos_id:
                    break
                cleaned.append(token)
            decoded_outputs.append(self.tokenizer.decode([cleaned])[0])

        return decoded_outputs
        

    def extract_answer(self): return

    def eval(self):
        with open(f"{self.save_path}/{self.save_name}", "w") as file:
            for index, batch in enumerate(self.dataloader):
                outputs = self.generate(batch["original"])
                print(outputs)
                break
                


if __name__ == "__main__":
    vocab = {
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

    tokenizer_config = {
        "vocab": vocab,
        "eos_id": 12,
        "padding_id": 13,
    }
    model_config = {
        "num_layers": 10,
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4,
        "vocab_size": 14,
        "max_seq_len": 30
    }
    dataset_config = {
        "num_samples": 10_000,
        "num_digits": [4, 5],
        "seed": 1001,
        "mode": "eval"
    }
    trainer_config = {
        "batch_size": 100,
        "device": "cuda",
        "model_config": model_config,
        "tokenizer_config": tokenizer_config,
        "dataset_config": dataset_config,
        "checkpoint_path": "/home/allan/nvim/generalization/checkpoints/generalized/model.pth",
        "save_path": "/home/allan/nvim/generalization/results/overfit/",
        "save_name": "overfit.jsonl"
    }
    evaler = AdditionEval(**trainer_config)
    evaler.eval()
