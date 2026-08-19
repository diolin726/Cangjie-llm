from .model import LLM
from .tokenization import tokenizer

__all__ = ["LLM", "main", "tokenizer"]


def main():
    from .training import main as training_main

    return training_main()
