import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Set the style for a clean, modern look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def create_smoothing_visualization():
    """
    Create a visualization explaining how smoothing works in n-gram models.
    """
    # Example counts for a specific context
    context = "the"
    next_words = ["cat", "dog", "mouse", "bird", "elephant", "unseen"]
    counts = [10, 5, 3, 1, 0, 0]  # "elephant" and "unseen" have zero counts
    
    # Calculate probabilities with and without smoothing
    total_count = sum(counts)
    
    # No smoothing (Maximum Likelihood Estimation)
    mle_probs = [count/total_count if count > 0 else 0 for count in counts]
    
    # Laplace (add-1) smoothing
    alpha = 1
    laplace_probs = [(count + alpha)/(total_count + alpha*len(counts)) for count in counts]
    
    # Add-0.1 smoothing
    alpha_small = 0.1
    add_small_probs = [(count + alpha_small)/(total_count + alpha_small*len(counts)) for count in counts]
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Title
    plt.suptitle("Understanding Smoothing in N-gram Models", fontsize=20, y=0.98)
    
    # Bar chart comparing probabilities
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(next_words))
    width = 0.25
    
    ax1.bar(x - width, mle_probs, width, label='No Smoothing (MLE)', color='#ff9999')
    ax1.bar(x, laplace_probs, width, label='Laplace (add-1) Smoothing', color='#66b3ff')
    ax1.bar(x + width, add_small_probs, width, label='Add-0.1 Smoothing', color='#99ff99')
    
    ax1.set_ylabel('Probability P(word|"the")')
    ax1.set_title('Effect of Smoothing on Probability Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(next_words)
    ax1.legend()
    
    # Highlight the zero-count problem
    for i, count in enumerate(counts):
        if count == 0:
            ax1.annotate('Zero count problem!', 
                        xy=(i - width, 0.01), 
                        xytext=(i - 1.5*width, 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Smoothing formula explanation
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    formula_text = r"""
    $\mathbf{Smoothing\ Formulas:}$
    
    $\mathbf{MLE:}$
    $P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}$
    
    $\mathbf{Laplace\ (Add-\alpha):}$
    $P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i) + \alpha}{count(w_{i-1}) + \alpha \cdot |V|}$
    
    Where:
    - $w_i$ is the current word
    - $w_{i-1}$ is the previous word (context)
    - $|V|$ is the vocabulary size
    - $\alpha$ is the smoothing parameter
    """
    ax2.text(0.5, 0.5, formula_text, ha='center', va='center', fontsize=14, 
             bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
    
    # Smoothing benefits and trade-offs
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    benefits_text = """
    Benefits of Smoothing:
    
    1. Handles unseen n-grams (zero counts)
    2. Prevents zero probabilities
    3. Improves model generalization
    4. Reduces overfitting to training data
    
    Trade-offs:
    
    1. Reduces probability of observed events
    2. Different α values work better for 
       different data sizes
    3. Simple smoothing may not capture 
       linguistic patterns effectively
    """
    ax3.text(0.5, 0.5, benefits_text, ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('smoothing_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_perplexity_visualization():
    """
    Create a visualization explaining perplexity in language models.
    """
    # Example perplexity values for different models
    models = ['Unigram', 'Bigram', 'Trigram', '4-gram', '5-gram']
    perplexities = [942, 170, 109, 82, 77]
    
    # Example probabilities assigned to a test sentence
    sentence = "The cat sat on the mat"
    tokens = sentence.split()
    
    # Made-up log probabilities for demonstration
    log_probs = {
        'Unigram': [-2.3, -4.1, -3.8, -2.5, -2.3, -4.2],  # Higher is worse
        'Bigram': [-1.8, -2.2, -2.0, -1.9, -1.7, -2.1],
        'Trigram': [-1.5, -1.8, -1.7, -1.6, -1.4, -1.9]
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Title
    plt.suptitle("Understanding Perplexity in Language Models", fontsize=20, y=0.98)
    
    # Perplexity vs. n-gram order
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(models, perplexities, 'o-', linewidth=2, markersize=10, color='#4285F4')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Perplexity (lower is better)')
    ax1.set_title('Perplexity vs. N-gram Order')
    ax1.grid(True)
    
    # Annotate the improvement
    for i in range(1, len(models)):
        improvement = perplexities[i-1] - perplexities[i]
        ax1.annotate(f'-{improvement}', 
                    xy=((i-1+i)/2, (perplexities[i-1]+perplexities[i])/2),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    
    # Log probabilities visualization
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(tokens))
    width = 0.25
    
    # Plot log probabilities (negative, so lower bars are better)
    ax2.bar(x - width, [-p for p in log_probs['Unigram']], width, label='Unigram', color='#DB4437')
    ax2.bar(x, [-p for p in log_probs['Bigram']], width, label='Bigram', color='#F4B400')
    ax2.bar(x + width, [-p for p in log_probs['Trigram']], width, label='Trigram', color='#0F9D58')
    
    ax2.set_ylabel('Log Probability (higher is better)')
    ax2.set_title('Log Probabilities Assigned to Test Sentence')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens)
    ax2.legend()
    
    # Perplexity formula and explanation
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    perplexity_text = r"""
    $\mathbf{Perplexity\ Formula:}$
    
    $PP(W) = \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}} = \sqrt[N]{\prod_{i=1}^{N}\frac{1}{P(w_i|w_{i-n+1}...w_{i-1})}} = 2^{-\frac{1}{N}\sum_{i=1}^{N}\log_2 P(w_i|w_{i-n+1}...w_{i-1})}$
    
    Where:
    - $W$ is the test text
    - $N$ is the number of tokens
    - $P(w_i|w_{i-n+1}...w_{i-1})$ is the probability of word $w_i$ given its context
    
    $\mathbf{Interpreting\ Perplexity:}$
    
    • Perplexity can be interpreted as the weighted average branching factor of a language model
    • Lower perplexity means the model is more confident and accurate in its predictions
    • A perplexity of k means the model is as confused as if it had to choose uniformly among k options for each word
    • As context length (n) increases, perplexity typically decreases, but with diminishing returns
    • Perplexity is affected by vocabulary size, corpus domain, and smoothing techniques
    """
    
    ax3.text(0.5, 0.5, perplexity_text, ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('perplexity_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ngram_sparsity_visualization():
    """
    Create a visualization explaining the sparsity problem in n-gram models.
    """
    # Example data: percentage of n-grams seen in training vs. n
    n_values = [1, 2, 3, 4, 5]
    coverage_percentages = [95, 60, 30, 12, 4]  # Made-up values for illustration
    
    # Number of possible n-grams with vocab size V
    vocab_size = 10000
    possible_ngrams = [vocab_size**n for n in n_values]
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Title
    plt.suptitle("The Sparsity Problem in N-gram Models", fontsize=20, y=0.98)
    
    # Coverage percentage vs. n
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(n_values, coverage_percentages, 'o-', linewidth=2, markersize=10, color='#4285F4')
    ax1.set_xlabel('N-gram Order (n)')
    ax1.set_ylabel('% of Possible N-grams Observed in Training')
    ax1.set_title('N-gram Coverage Decreases with n')
    ax1.set_ylim(0, 100)
    ax1.grid(True)
    
    # Number of possible n-grams vs. n (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(n_values, possible_ngrams, 'o-', linewidth=2, markersize=10, color='#DB4437')
    ax2.set_xlabel('N-gram Order (n)')
    ax2.set_ylabel('Number of Possible N-grams (log scale)')
    ax2.set_title('Possible N-grams Grow Exponentially with n')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    # Illustration of the sparsity problem
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create a visual representation of sparsity
    grid_size = 20
    sparsity_levels = [0.05, 0.4, 0.7, 0.88, 0.96]  # Corresponds to coverage_percentages
    
    for i, sparsity in enumerate(sparsity_levels):
        if i >= len(n_values):
            break
            
        # Create a grid
        grid = np.random.random((grid_size, grid_size)) > sparsity
        
        # Plot the grid
        ax_sub = plt.subplot(gs[1, 0], position=[0.1 + i*0.15, 0.1, 0.12, 0.3])
        ax_sub.imshow(grid, cmap='Blues', interpolation='nearest')
        ax_sub.set_title(f'{n_values[i]}-gram')
        ax_sub.axis('off')
    
    # Add explanation
    explanation_text = """
    The Sparsity Problem:
    
    As n increases:
    • The number of possible n-grams grows exponentially (V^n)
    • The percentage of n-grams observed in training decreases rapidly
    • Most possible n-grams are never seen in training data
    
    This leads to:
    • Zero probability estimates for many valid n-grams
    • Poor generalization to new text
    • Need for smoothing and backoff techniques
    
    Solutions:
    • Smoothing: Assign non-zero probabilities to unseen n-grams
    • Backoff: Use shorter contexts when longer ones aren't available
    • Interpolation: Combine probabilities from different order n-grams
    """
    
    ax_text = plt.subplot(gs[1, 1])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, explanation_text, ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('ngram_sparsity_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create all visualizations
    create_smoothing_visualization()
    create_perplexity_visualization()
    create_ngram_sparsity_visualization()
    print("All visualizations created successfully!")
