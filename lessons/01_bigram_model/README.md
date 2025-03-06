# Lesson 1: Bigram Language Model

## Overview

The goal of this lesson is to teach you about (one of the) most simple types of language models, the bigram. Inside of explore_bigrams.ipynb, we explore
how to implment bigrams, as well as the underlying structure that makes them tick. 

## Learning Objectives

By the end of this lesson, you will:

1. Understand the statistical foundations of language modeling
2. Implement a bigram model from scratch using PyTorch
3. Learn about probability estimation and sampling techniques
4. Generate text using a trained bigram model
5. Evaluate the limitations of simple statistical models - yes, bigrams kind of suck

## Prerequisites (please let me know if you want a lesson 0, covering these basics)

- Basic Python programming knowledge
- Understanding of probability concepts
- Familiarity with data processing

## Setup

1. Install the required dependencies:
   ```bash
   pip install datasets torch numpy matplotlib pandas tqdm
   ```

2. Explore interactively:
   ```bash
   jupyter notebook explore_bigrams.ipynb
   ```

## Code Walkthrough

The implementation consists of several key components:

### 1. Data Loading and Preprocessing

We a modified version of the [ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) containing scientific paper summaries. The data is loaded, preprocessed, and split into training and testing sets.

**Why it matters**: Having high quality training data leads to high quality models, as with many avenues - garbage in, garbage out.

### 2. Tokenization

Text is converted into numerical tokens that the model can process. For our character-level model, each unique character is assigned an integer ID.

**How it works**:
- We create a vocabulary of all unique characters in the dataset
- We build mapping dictionaries to convert between characters and their numerical IDs
- The `encode()` function converts text to token IDs
- The `decode()` function converts token IDs back to text

**Why it matters**: Tokenization is the foundation of all language models, transforming human-readable text into machine-processable numbers. We will dig deeper into to tokenization in later lessons (as normally, tokens are more than one character), or see [Tiktokenizer](https://tiktokenizer.vercel.app/).

### 3. Batch Generation

We create training batches by sampling random sequences from our dataset, along with their corresponding "next character" targets.

**How it works**:
- We define a context window size (block_size) that determines how many characters the model sees at once
- For each input sequence, the target is the same sequence shifted one position to the right
- Multiple sequences are grouped into batches for efficient training

**Why it matters**: Proper batch generation ensures the model sees diverse examples and trains efficiently.

### 4. Model Architecture

Our BigramLanguageModel class implements a simple neural network that predicts the next character based on the current one.

**How it works**:
- The model contains a single learnable parameter: a token embedding table
- This table can be viewed as a matrix of transition probabilities between characters
- The forward pass looks up embeddings for each input token
- The generate method samples new tokens based on the model's predictions

**Why it matters**: Even this simple architecture demonstrates the core principles behind more complex language models.

### 5. Training Loop

We optimize the model parameters using gradient descent to minimize the cross-entropy loss between predictions and actual next characters.

**How it works**:
- We define a loss function that measures prediction accuracy
- We use the AdamW optimizer to update model parameters
- We periodically evaluate progress and generate sample text

**Why it matters**: The training process is where the model learns the statistical patterns in the data.

### 6. Text Generation

After training, we use the model to generate new text by repeatedly sampling from its predicted probability distributions.

**How it works**:
- Start with a seed token (or empty context)
- Get the model's prediction for the next token
- Sample from this probability distribution
- Add the sampled token to the context and repeat

**Why it matters**: Text generation demonstrates what the model has learned and highlights its capabilities and limitations.

## Key Concepts

- **Markov Assumption**: The probability of a token depends only on the previous token, not on any earlier context.
- **Maximum Likelihood Estimation**: Estimating probabilities based on observed frequencies in the training data.
- **Sampling**: Generating new tokens by randomly selecting from the model's predicted probability distribution.
- **Sparsity Problem**: The challenge of estimating probabilities for rare or unseen events.
- **Cross-Entropy Loss**: A measure of the difference between predicted probability distributions and actual outcomes.

## Exercises

1. **Modify the context window size**: Change the `block_size` parameter and observe how it affects training and generation. Does a larger context window improve the model's output?

2. **Implement word-level tokenization**: Modify the code to work with words instead of characters. How does this change the model's behavior?

3. **Add temperature control**: Implement a temperature parameter in the `generate` method to control the randomness of the generated text.

4. **Visualize the learned probabilities**: Create a heatmap visualization of the model's learned transition probabilities for common characters.

5. **Implement perplexity calculation**: Add a method to evaluate the model's performance using perplexity, a standard metric for language models.

## Limitations of Bigram Models

Bigram models have several inherent limitations:

1. **Limited context**: They only consider the immediately preceding token, ignoring all earlier context.
2. **No long-range dependencies**: They cannot capture relationships between distant parts of text.
3. **Lack of semantic understanding**: They model statistical patterns without understanding meaning.
4. **Repetitive patterns**: Generated text often contains repetitive sequences or gets stuck in loops.

## Further Reading

1. Jurafsky, D., & Martin, J. H. (2009). Speech and Language Processing (2nd Edition). Chapter 4: N-grams.
2. Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing. MIT Press.
3. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling.
4. Bengio, Y., et al. (2003). A Neural Probabilistic Language Model. Journal of Machine Learning Research.

## Next Steps

After completing this lesson, you'll be ready to move on to Lesson 2, where we'll extend the bigram model to higher-order n-grams and explore more sophisticated smoothing techniques to handle the sparsity problem. 