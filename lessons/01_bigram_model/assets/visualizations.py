import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrowPatch

# Set the style for a clean, modern look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# -------------------- TOKENIZATION DIAGRAM -------------------- #
def create_tokenization_diagram():
    """Create the original tokenization diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    text = "We study language models"
    
    chars = sorted(list(set(text + " ")))
    str_to_i = {ch: i for i, ch in enumerate(chars)}
    
    tokens = [str_to_i[ch] for ch in text]
    for i, char in enumerate(text):
        # Add character with box
        ax.add_patch(Rectangle((i, 0.6), 0.9, 0.9, fill=True, color='#e0f0ff', alpha=0.5))
        ax.text(i + 0.45, 1.05, char, ha='center', va='center', fontsize=12)
    
    for i, token in enumerate(tokens):
        # Add token with box
        ax.add_patch(Rectangle((i, -0.3), 0.9, 0.9, fill=True, color='#f0e0ff', alpha=0.5))
        ax.text(i + 0.45, 0.15, str(token), ha='center', va='center', fontsize=12)
    
    # Add arrows connecting characters to tokens
    for i in range(len(text)):
        arrow = FancyArrowPatch((i + 0.45, 0.7), (i + 0.45, 0.3), 
                              arrowstyle='-|>', mutation_scale=12,
                              color='#007acc', linewidth=1)
        ax.add_patch(arrow)
    
    # Add vocabulary display on the side
    vocab_x = len(text) + 1.5
    ax.add_patch(Rectangle((vocab_x, -0.3), 3.5, 2, fill=True, color='#f5f5f5', alpha=0.5))
    ax.text(vocab_x + 1.75, 1.5, "Vocabulary", ha='center', va='center', fontsize=12, fontweight='bold')
    
    for i, (char, idx) in enumerate(sorted(str_to_i.items())):
        y_pos = 1.2 - (i * 0.2)
        if y_pos < -0.2:
            continue  # Skip if we run out of space
        ax.text(vocab_x + 0.4, y_pos, f"'{char}':", ha='left', va='center', fontsize=10)
        ax.text(vocab_x + 2.2, y_pos, f"{idx}", ha='left', va='center', fontsize=10)
    
    # Add title and labels
    ax.text(len(text)/2, 1.8, "Character-level Tokenization", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(len(text)/2, -0.8, "Token IDs", ha='center', va='center', fontsize=12)
    ax.text(len(text)/2, 1.6, "Original Text", ha='center', va='center', fontsize=12)
    
    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, vocab_x + 4)
    ax.set_ylim(-1, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('tokenization_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_bigram_diagram():
    """Create a simple diagram illustrating how a bigram model works."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), 
                                  gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Example context and next character
    context = "th"
    next_char = "e"
    
    # --- Left panel: Context and prediction ---
    # Draw the context
    for i, char in enumerate(context):
        ax1.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, 
                                   fill=True, color='lightblue', alpha=0.7))
        ax1.text(i + 0.45, 0.45, char, ha='center', va='center', fontsize=16)
    
    # Highlight the last character (current token)
    ax1.add_patch(plt.Rectangle((len(context)-1, 0), 0.9, 0.9, 
                               fill=True, color='gold', alpha=0.7, 
                               edgecolor='black', linewidth=2))
    
    # Add arrow to next character
    arrow = FancyArrowPatch((len(context)-0.1, 0.45), (len(context)+0.6, 0.45), 
                           arrowstyle='->', color='red', linewidth=2)
    ax1.add_patch(arrow)
    
    # Add the predicted character
    ax1.add_patch(plt.Rectangle((len(context)+0.7, 0), 0.9, 0.9, 
                               fill=True, color='lightgreen', alpha=0.7))
    ax1.text(len(context)+1.15, 0.45, next_char, ha='center', va='center', fontsize=16)
    
    # Add labels
    ax1.text(len(context)/2, 1.3, "Bigram Model Prediction", 
             ha='center', fontsize=14, fontweight='bold')
    ax1.text(len(context)-0.55, -0.3, "Current\nToken", ha='center', fontsize=12)
    ax1.text(len(context)+1.15, -0.3, "Predicted\nToken", ha='center', fontsize=12)
    
    # --- Right panel: Probability distribution ---
    # Example probabilities for the next character given 'h'
    probs = {'e': 0.5, 'a': 0.2, 'i': 0.15, 'o': 0.1, 'u': 0.05}
    
    # Sort by probability
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    chars = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create bar chart
    bars = ax2.bar(chars, values, color=['gold' if c == next_char else 'lightblue' for c in chars])
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f"{val:.2f}", ha='center', fontsize=11)
    
    # Add title and labels
    ax2.set_title("Probability Distribution\nafter seeing 'h'", fontsize=14)
    ax2.set_xlabel("Next Character", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(0, 0.6)  # Give some space for the value labels
    
    # Add explanation
    ax1.text(0, -1, "The bigram model predicts the next character based on the current character.", 
             ha='left', fontsize=12)
    ax1.text(0, -1.3, f"Given '{context[-1]}', the model assigns highest probability to '{next_char}'.", 
             ha='left', fontsize=12)
    
    # Clean up the plots
    ax1.set_xlim(-0.5, len(context) + 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis('off')
    
    plt.tight_layout()
    plt.savefig('bigram_model_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_loss_minimization_diagram():
    """Create a visualization explaining what loss minimization means for a bigram model."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Example character and its possible next characters
    current_char = 't'
    next_chars = ['h', 'e', 'a', 'o', 'i', ' ', 'r']
    
    # Ground truth distribution (from data)
    true_probs = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02])
    
    # Model predictions at different training stages
    initial_probs = np.array([0.14, 0.15, 0.13, 0.14, 0.15, 0.14, 0.15])  # Random (high loss)
    mid_probs = np.array([0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.07])      # Partially trained
    final_probs = np.array([0.33, 0.24, 0.16, 0.11, 0.09, 0.05, 0.02])    # Well trained (low loss)
    
    # Calculate cross-entropy loss for each stage
    def cross_entropy(pred, true):
        # Add small epsilon to avoid log(0)
        pred = np.clip(pred, 1e-10, 1.0)
        return -np.sum(true * np.log(pred))
    
    initial_loss = cross_entropy(initial_probs, true_probs)
    mid_loss = cross_entropy(mid_probs, true_probs)
    final_loss = cross_entropy(final_probs, true_probs)
    
    # Left plot: Bar chart comparing distributions
    x = np.arange(len(next_chars))
    width = 0.2
    
    # Plot the bars
    ax1.bar(x - width, true_probs, width, label='True Distribution', color='#2ca02c', alpha=0.7)
    ax1.bar(x, initial_probs, width, label=f'Initial Model (Loss: {initial_loss:.2f})', color='#d62728', alpha=0.7)
    ax1.bar(x + width, final_probs, width, label=f'Trained Model (Loss: {final_loss:.2f})', color='#1f77b4', alpha=0.7)
    
    # Add labels and title
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Probability Distribution After Character "{current_char}"')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"'{c}'" for c in next_chars])
    ax1.legend()
    
    # Right plot: Loss curve during training
    training_steps = np.arange(0, 101)
    # Simulate exponentially decreasing loss
    losses = 2.0 * np.exp(-0.03 * training_steps) + 0.5
    
    # Mark our three example points
    example_steps = [0, 40, 100]
    example_losses = [initial_loss, mid_loss, final_loss]
    
    # Plot the loss curve
    ax2.plot(training_steps, losses, color='#ff7f0e', linewidth=2)
    ax2.scatter(example_steps, example_losses, color=['#d62728', '#ff7f0e', '#1f77b4'], s=100, zorder=5)
    
    # Add annotations for the example points
    ax2.annotate('Initial (Random)', xy=(0, initial_loss), xytext=(5, initial_loss+0.2),
                arrowprops=dict(arrowstyle='->'))
    ax2.annotate('Partially Trained', xy=(40, mid_loss), xytext=(45, mid_loss+0.2),
                arrowprops=dict(arrowstyle='->'))
    ax2.annotate('Well Trained', xy=(100, final_loss), xytext=(80, final_loss+0.2),
                arrowprops=dict(arrowstyle='->'))
    
    # Add labels and title
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss (Cross-Entropy)')
    ax2.set_title('Loss Minimization During Training')
    ax2.set_ylim(0, 3)
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             "Minimizing loss means making the model's predicted probabilities closer to the true distribution.\n"
             "Lower loss = better predictions = more realistic text generation.", 
             ha='center', fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('loss_minimization_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create all diagrams
    create_tokenization_diagram()
    create_bigram_diagram()
    create_loss_minimization_diagram()
    print("Diagrams created successfully!")
