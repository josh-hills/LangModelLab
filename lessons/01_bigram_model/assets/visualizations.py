import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrowPatch

# Set the style for a clean, modern look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# -------------------- TOKENIZATION DIAGRAM -------------------- #
def create_tokenization_diagram():
    """Create the original tokenization diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    text = "We love language models"
    
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

if __name__ == "__main__":
    # Create both diagrams
    create_tokenization_diagram()
    create_bigram_diagram()
    print("Diagrams created successfully!")
