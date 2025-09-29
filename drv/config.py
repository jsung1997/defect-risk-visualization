
from dataclasses import dataclass

@dataclass
class DRVConfig:
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    top_k: int = 10  # for ranking evaluation
    output_dir: str = "outputs"
