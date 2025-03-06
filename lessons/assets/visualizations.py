import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 6))
text = "We study language models"

chars = sorted(list(set(text + " ")))
str_to_i = {ch: i for i, ch in enumerate(chars)}

tokens = [str_to_i[ch] for ch in text]
for i, char in enumerate(text):
    # Add character with box
    ax.add_patch(Rectangle((i, 0.6), 0.9, 0.9, fill=True, color='#e0f0ff', alpha=0.5))
    ax.text(i + 0.45, 1.05, char, ha='center', va='center', fontsize=14)

for i, token in enumerate(tokens):
    # Add token with box
    ax.add_patch(Rectangle((i, -0.3), 0.9, 0.9, fill=True, color='#f0e0ff', alpha=0.5))
    ax.text(i + 0.45, 0.15, str(token), ha='center', va='center', fontsize=14)

# Add arrows connecting characters to tokens
for i in range(len(text)):
    arrow = FancyArrowPatch((i + 0.45, 0.7), (i + 0.45, 0.3), 
                          arrowstyle='-|>', mutation_scale=15, 
                          color='#007acc', linewidth=1.5)
    ax.add_patch(arrow)

# Add vocabulary display on the side
vocab_x = len(text) + 1.5
ax.add_patch(Rectangle((vocab_x, -0.3), 3.5, 2, fill=True, color='#f5f5f5', alpha=0.5))
ax.text(vocab_x + 1.75, 1.5, "Vocabulary", ha='center', va='center', fontsize=14, fontweight='bold')

for i, (char, idx) in enumerate(sorted(str_to_i.items())):
    y_pos = 1.2 - (i * 0.2)
    if y_pos < -0.2:
        continue  # Skip if we run out of space
    ax.text(vocab_x + 0.4, y_pos, f"'{char}':", ha='left', va='center', fontsize=12)
    ax.text(vocab_x + 2.2, y_pos, f"{idx}", ha='left', va='center', fontsize=12)

# Add title and labels
ax.text(len(text)/2, 1.8, "Character-level Tokenization", ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(len(text)/2, -0.8, "Token IDs", ha='center', va='center', fontsize=14)
ax.text(len(text)/2, 1.6, "Original Text", ha='center', va='center', fontsize=14)

# Set axis limits and remove ticks
ax.set_xlim(-0.5, vocab_x + 4)
ax.set_ylim(-1, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

plt.tight_layout()
plt.show()