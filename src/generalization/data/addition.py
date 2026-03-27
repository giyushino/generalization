import random


class AdditionDataset():
    def __init__(self, num_samples: int, num_digits: list[int]):
        assert len(num_digits) == 2, "num_digits must contain lower and upper bound"
        self.lower = int(f"1{'0' * (num_digits[0] - 1)}")
        self.upper = int(f"{'9' * num_digits[1]}")
        self.num_samples = num_samples
        self.data = self._populate(self.num_samples, self.lower, self.upper)

    def _populate(self, num_samples: int, lower: int, upper: int):
        data = []
        
        for _ in range(num_samples):
            num1 = random.randint(lower, upper)
            num2 = random.randint(lower, upper)
            equation = f"{num1}+{num2}={num1 + num2}" 
            data.append(equation)

        return data





if __name__ == "__main__":
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    
    dataset_config = {
        "num_samples": 100,
        "num_digits": [4, 5]
    }

    dataset = AdditionDataset(**dataset_config)
    print(dataset.data) 
