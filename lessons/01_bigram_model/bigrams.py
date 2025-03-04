"""
Bigram Language Model Implementation
"""

import os
import re
from datasets import load_dataset
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Type aliases for clarity
BigramCounts = Dict[Tuple[str, str], int]
BigramProbs = Dict[Tuple[str, str], float]
Vocabulary = Dict[str, int]

def load_arxiv_dataset(
    limit: Optional[int] = None,
    default_dir: str = "/home/josh/Documents/Code/FigureGPT/src/datasets",
    categories: Optional[List[str]] = None
) -> List[str]:
    
    print("Loading ArXiv dataset...")
    dataset = load_dataset("arxiv-community/arxiv_dataset", data_dir=default_dir)
    
    # Access the 'train' split which contains all the data
    data = dataset["train"]
    
    # Filter by categories if specified
    if categories:
        data = [item for item in data if any(cat in item["categories"] for cat in categories)]
    
    # Extract abstracts
    abstracts = [item["abstract"] for item in data[:limit] if item["abstract"]]
    
    print(f"Loaded {len(abstracts)} paper abstracts")
    return abstracts

# Rest of the code remains the same...
def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    
    # Remove special characters and digits, keeping only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Add special tokens for start and end of sequence
    text = "<s> " + text + " </s>"
    
    # Split by whitespace
    tokens = text.split()
    
    return tokens

class BigramModel:
    """
    Bigram language model that learns word transition probabilities.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize the bigram model.
        
        Args:
            smoothing: Laplace smoothing parameter (alpha)
        """
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: BigramCounts = defaultdict(int)
        self.bigram_probs: BigramProbs = {}
        self.vocab: Vocabulary = {}
        self.smoothing = smoothing
        self.is_trained = False
    
    def train(self, texts: List[str]) -> None:
        """
        Train the bigram model on a corpus of texts.
        
        Args:
            texts: List of text documents
        """
        print("Preprocessing texts...")
        processed_texts = [preprocess_text(text) for text in tqdm(texts)]
        
        print("Tokenizing and counting bigrams...")
        for text in tqdm(processed_texts):
            tokens = tokenize(text)
            
            # Count unigrams
            self.unigram_counts.update(tokens)
            
            # Count bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_counts[bigram] += 1
        
        # Build vocabulary
        self.vocab = {word: idx for idx, word in enumerate(self.unigram_counts.keys())}
        vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"Total bigrams: {len(self.bigram_counts)}")
        
        # Calculate bigram probabilities with Laplace smoothing
        print("Computing bigram probabilities...")
        for (w1, w2), count in tqdm(self.bigram_counts.items()):
            # P(w2|w1) = (count(w1,w2) + alpha) / (count(w1) + alpha*|V|)
            denominator = self.unigram_counts[w1] + self.smoothing * vocab_size
            self.bigram_probs[(w1, w2)] = (count + self.smoothing) / denominator
        
        self.is_trained = True
        print("Training complete!")
    
    def get_probability(self, w1: str, w2: str) -> float:
        """
        Get the probability of word w2 following word w1.
        
        Args:
            w1: First word
            w2: Second word
            
        Returns:
            Conditional probability P(w2|w1)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting probabilities")
        
        # If the bigram exists in our model
        if (w1, w2) in self.bigram_probs:
            return self.bigram_probs[(w1, w2)]
        
        # If w1 exists but the bigram doesn't, use smoothed probability
        if w1 in self.unigram_counts:
            vocab_size = len(self.vocab)
            return self.smoothing / (self.unigram_counts[w1] + self.smoothing * vocab_size)
        
        # If w1 doesn't exist, use uniform probability
        return 1.0 / len(self.vocab) if self.vocab else 0.0
    
    def generate_text(self, max_length: int = 50, start_word: str = "<s>") -> str:
        """
        Generate text using the trained bigram model.
        
        Args:
            max_length: Maximum number of words to generate
            start_word: Word to start generation with
            
        Returns:
            Generated text
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating text")
        
        if start_word not in self.vocab:
            start_word = "<s>"
        
        current_word = start_word
        generated_words = [current_word]
        
        for _ in range(max_length):
            # Find all possible next words and their probabilities
            next_word_probs = {
                w2: self.get_probability(current_word, w2)
                for w1, w2 in self.bigram_probs.keys()
                if w1 == current_word
            }
            
            if not next_word_probs:
                break
            
            # Convert to list of (word, prob) pairs for sampling
            words, probs = zip(*next_word_probs.items())
            
            # Normalize probabilities
            probs = np.array(probs) / sum(probs)
            
            # Sample next word
            next_word = np.random.choice(words, p=probs)
            
            # Stop if we reach the end token
            if next_word == "</s>":
                break
            
            generated_words.append(next_word)
            current_word = next_word
        
        # Remove start and end tokens for output
        clean_text = " ".join(
            [word for word in generated_words if word not in ["<s>", "</s>"]]
        )
        return clean_text

# Main function to demonstrate the model
def main():
    # Load a small subset of the ArXiv dataset for demonstration
    abstracts = load_arxiv_dataset(limit=1000)
    
    # Train the bigram model
    model = BigramModel(smoothing=0.1)
    model.train(abstracts)
    
    # Generate some sample text
    print("\nGenerated text samples:")
    for _ in range(5):
        generated_text = model.generate_text(max_length=30)
        print(f"- {generated_text}")
    
    # Example of calculating probabilities
    print("\nSample bigram probabilities:")
    word_pairs = [
        ("the", "model"),
        ("we", "propose"),
        ("in", "this"),
        ("neural", "network"),
        ("random", "word")
    ]
    
    for w1, w2 in word_pairs:
        prob = model.get_probability(w1, w2)
        print(f"P({w2}|{w1}) = {prob:.6f}")

if __name__ == "__main__":
    main()
    