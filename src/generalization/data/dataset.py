import random

class AdditionDataset():
    def __init__(self, num_samples: int, num_digits: list[int], seed: int, mode: str = "train"):
        assert len(num_digits) == 2, "num_digits must contain lower and upper bound"
        self.lower = int(f"1{'0' * (num_digits[0] - 1)}")
        self.upper = int(f"{'9' * num_digits[1]}")
        self.num_samples = num_samples
        if mode == "train":
            self.data = self._populate_train(self.num_samples, self.lower, self.upper)
        else:
            self.data = self._populate_eval(self.num_samples, self.lower, self.upper)
        random.seed(seed)

    def _populate_train(self, num_samples: int, lower: int, upper: int):
        data = []
        
        for _ in range(num_samples):
            num1 = random.randint(lower, upper)
            num2 = random.randint(lower, upper)
            equation = f"{num1}+{num2}={num1 + num2}" 
            data.append(equation)

        return data

    def _populate_eval(self, num_samples: int, lower: int, upper: int):
        data = []
        
        for _ in range(num_samples):
            num1 = random.randint(lower, upper)
            num2 = random.randint(lower, upper)
            equation = f"{num1}+{num2}="
            ground_truth = f"{num1 + num2}"
            data.append({"question": equation, "ground_truth": ground_truth})

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: slice | int):
        return self.data[key]



if __name__ == "__main__":
    dataset_config = {
        "num_samples": 100,
        "num_digits": [4, 5],
        "seed": 42,
        "mode": eval
    }

    dataset = AdditionDataset(**dataset_config)
    print(dataset.data[0]["question"])
