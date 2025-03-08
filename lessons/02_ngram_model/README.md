# Lesson 2: N-gram Language Models

## Overview

This lesson builds on the foundations established in Lesson 1 (Bigram Model) by extending to higher-order n-gram models. While bigram models only consider the immediately preceding token, n-gram models can use multiple previous tokens as context, allowing for more sophisticated language modeling.

## Learning Objectives

By the end of this lesson, you will:

1. Understand how n-gram models extend bigram models to capture more context
2. Implement a variable-order n-gram model from scratch using PyTorch
3. Apply smoothing techniques to handle unseen n-grams
4. Compare the performance of different n-gram orders (n=2, 3, 4, etc.)
5. Evaluate language models using perplexity
6. Visualize n-gram probabilities to gain insights into what the model has learned

## Prerequisites

- Completion of Lesson 1: Bigram Language Model
- Basic Python programming knowledge
- Understanding of probabilities and basic statistics

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Explore interactively:
   ```bash
   jupyter notebook ngram.py
   ```

## Code Walkthrough

The implementation extends the bigram model with several key improvements:

### 1. Variable N-gram Order

Unlike the bigram model which always uses exactly one token of context, our n-gram model allows for configurable context length through the `n` parameter.

**How it works**:
- The model is initialized with a specified value of `n` (default is 3 for trigrams)
- For n=3, the model predicts the next token based on the previous 2 tokens
- The context size is always n-1 tokens

**Why it matters**: Larger values of n capture more context, potentially leading to more coherent text generation, but also face increased data sparsity issues.

### 2. Two Implementation Approaches

We provide two different implementations of n-gram models:

#### Neural N-gram Model

Similar to the neural bigram model from Lesson 1, but extended to handle multiple context tokens.

**How it works**:
- Uses separate embedding tables for each position in the context
- Concatenates the embeddings and applies a linear transformation
- Outputs logits for the next token prediction

#### Count-Based N-gram Model with Smoothing

A traditional statistical approach that explicitly counts n-gram occurrences.

**How it works**:
- Counts occurrences of all n-grams in the training data
- Applies Laplace smoothing to handle unseen n-grams
- Calculates probabilities as (count + α) / (context_count + α * vocab_size)

**Why it matters**: The count-based approach provides more interpretable results and allows for visualization of learned probabilities, while the neural approach can be more efficient for large vocabularies.

### 3. Smoothing Techniques

A critical improvement over the basic bigram model is the addition of smoothing to handle unseen n-grams.

**How it works**:
- Laplace (add-α) smoothing adds a small constant α to all counts
- This ensures that unseen n-grams receive a small but non-zero probability
- The α parameter controls the strength of smoothing

**Why it matters**: As n increases, the number of possible n-grams grows exponentially, making it impossible to observe all valid combinations in training data. Smoothing prevents the model from assigning zero probability to valid but unseen sequences.

### 4. Backoff for Unseen Contexts

When generating text, we implement a backoff strategy for handling unseen contexts.

**How it works**:
- If the current context has never been seen in training, try a shorter context
- Continue backing off until a known context is found or default to random selection

**Why it matters**: Backoff strategies allow the model to gracefully handle novel situations by falling back to more general knowledge.

### 5. Perplexity Evaluation

We introduce perplexity as a standard metric for evaluating language models.

**How it works**:
- Calculate the average log probability that the model assigns to each token in a test set
- Perplexity is 2 raised to the negative of this average log probability
- Lower perplexity indicates better prediction accuracy

**Why it matters**: Perplexity provides a quantitative measure of model performance, allowing for objective comparison between different models or configurations.

### 6. Visualization Tools

We provide tools to visualize what the model has learned.

**How it works**:
- Plot the top predicted next tokens for common contexts
- Compare perplexity across different values of n

**Why it matters**: Visualizations help build intuition about how n-gram models work and what patterns they capture.

## Key Concepts

- **N-gram Model**: A language model that predicts the next token based on the previous n-1 tokens.
- **Context Size**: The number of previous tokens considered when making a prediction (n-1).
- **Data Sparsity**: The problem of insufficient observations for many possible n-grams, which worsens as n increases.
- **Smoothing**: Techniques to assign non-zero probabilities to unseen n-grams.
- **Backoff**: Strategies for handling unseen contexts by using shorter contexts.
- **Perplexity**: A measure of how well a probability model predicts a sample, with lower values indicating better performance.

## The N-gram Trade-off

N-gram models present a fundamental trade-off:

- **Higher n**: Captures more context, potentially leading to more coherent text generation
- **Lower n**: Suffers less from data sparsity, requiring less training data

As n increases:
1. The model can capture longer patterns and dependencies
2. The number of possible n-grams grows exponentially (vocab_size^n)
3. The proportion of n-grams seen in training decreases
4. Memory requirements increase substantially
5. Smoothing and backoff become increasingly important

## Exercises

1. **Experiment with different values of n**: Try n=2, 3, 4, 5 and compare the generated text and perplexity. At what point does increasing n stop improving performance?

2. **Implement different smoothing techniques**: Try implementing Good-Turing or Kneser-Ney smoothing and compare their performance with Laplace smoothing.

3. **Analyze the effect of the smoothing parameter α**: Try different values of α (0.001, 0.01, 0.1, 1.0) and observe how they affect perplexity and generation.

4. **Implement a more sophisticated backoff strategy**: Modify the model to use interpolation between different order n-grams instead of simple backoff.

5. **Compare character-level vs. word-level n-grams**: Modify the code to work with words instead of characters and compare the results.

## Limitations of N-gram Models

Despite their improvements over bigram models, n-gram models still have several limitations:

1. **Fixed context window**: They can only consider a fixed number of previous tokens, regardless of their importance.
2. **No generalization**: They treat each unique n-gram as a separate entity, without recognizing similarities between related words or phrases.
3. **Memory requirements**: Storing counts for all observed n-grams becomes prohibitive as n increases.
4. **Data sparsity**: Even with smoothing, higher-order n-grams suffer from insufficient observations.
5. **No semantic understanding**: They model statistical patterns without understanding meaning.

## Resources

1. Jurafsky, D., & Martin, J. H. (2009). [Speech and Language Processing Chapter 4: N-grams.](https://web.stanford.edu/~jurafsky/slp3/4.pdf)
2. Chen, S. F., & Goodman, J. (1999). [An empirical study of smoothing techniques for language modeling.](https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf)
3. Kneser, R., & Ney, H. (1995). [Improved backing-off for m-gram language modeling.](https://www.ee.columbia.edu/~stanchen/spring16/e6884/papers/KneserNey95.pdf)

## Next Steps

After completing this lesson, you'll be ready to move on to Lesson 3, where we'll introduce word embeddings as a way to overcome the limitations of n-gram models by representing words in a continuous vector space that captures semantic relationships. 