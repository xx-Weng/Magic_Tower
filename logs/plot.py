import matplotlib.pyplot as plt
import re
import os
import matplotlib.gridspec as gridspec
import math
import numpy as np

# Get the directory where the current script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Collect all txt files and categorize them
show_files = []
no_show_files = []

for filename in os.listdir(base_dir):
    if filename.endswith('.txt') and filename != 'plot.py':
        if 'show' in filename.lower():
            show_files.append(filename)
        else:
            no_show_files.append(filename)

# Sort by model name (assuming model name is in the filename)
def get_model_name(filename):
    # Remove .txt extension and show-related suffixes
    name = filename.replace('.txt', '').replace('_show', '').replace('show_', '')
    return name

# Sort files
show_files.sort(key=get_model_name)
no_show_files.sort(key=get_model_name)

# Get all unique model names
all_models = set()
for filename in show_files + no_show_files:
    all_models.add(get_model_name(filename))
all_models = sorted(list(all_models))

if not all_models:
    print("No txt files found")
    exit()

# Create 3 rows and multiple columns layout
cols = len(all_models)
rows = 3

# Create large figure
fig = plt.figure(figsize=(6 * cols, 5 * rows))
gs = gridspec.GridSpec(rows, cols, figure=fig)

# Create mapping from model names to files
show_dict = {get_model_name(f): f for f in show_files}
no_show_dict = {get_model_name(f): f for f in no_show_files}

# Store statistics data for the third row bar charts
stats_data = {
    'show': {'continue_rate': [], 'run_rate': [], 'continue_high_wr': [], 'run_low_wr': []},
    'no_show': {'continue_rate': [], 'run_rate': [], 'continue_high_wr': [], 'run_low_wr': []}
}

# Process each model
for col_idx, model_name in enumerate(all_models):
    # First row: show files
    if model_name in show_dict:
        filename = show_dict[model_name]
        txt_path = os.path.join(base_dir, filename)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        win_rates = []
        hp_ratios = []
        choices = []
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            m1 = re.search(r'Win Rate: ([\d\.]+), Choice: ([cr])', lines[i])
            m2 = re.search(r'HP ([\d\.]+)/([\d\.]+)', lines[i+1])
            if m1 and m2:
                win_rate = float(m1.group(1))
                choice = m1.group(2)
                hp = float(m2.group(1))
                max_hp = float(m2.group(2))
                hp_ratio = hp / max_hp if max_hp > 0 else 0
                win_rates.append(win_rate)
                hp_ratios.append(hp_ratio)
                choices.append(choice)

        # Calculate statistics
        total_points = len(choices)
        continue_count = choices.count('c')
        run_count = choices.count('r')
        continue_ratio = continue_count / total_points if total_points > 0 else 0
        run_ratio = run_count / total_points if total_points > 0 else 0
        
        # Continue rate when win rate > 0.5
        high_wr_choices = [ch for ch, wr in zip(choices, win_rates) if wr > 0.5]
        continue_high_wr_ratio = high_wr_choices.count('c') / len(high_wr_choices) if high_wr_choices else 0
        
        # Run rate when win rate < 0.5
        low_wr_choices = [ch for ch, wr in zip(choices, win_rates) if wr < 0.5]
        run_low_wr_ratio = low_wr_choices.count('r') / len(low_wr_choices) if low_wr_choices else 0
        
        # Store statistics data
        stats_data['show']['continue_rate'].append(continue_ratio)
        stats_data['show']['run_rate'].append(run_ratio)
        stats_data['show']['continue_high_wr'].append(continue_high_wr_ratio)
        stats_data['show']['run_low_wr'].append(run_low_wr_ratio)

        # Create subplot (first row)
        ax = fig.add_subplot(gs[0, col_idx])

        # Plot points
        c_points = [(hr, wr) for hr, wr, ch in zip(hp_ratios, win_rates, choices) if ch == 'c']
        r_points = [(hr, wr) for hr, wr, ch in zip(hp_ratios, win_rates, choices) if ch == 'r']
        
        if c_points:
            c_hrs, c_wrs = zip(*c_points)
            ax.scatter(c_hrs, c_wrs, c='red', label='c (Continue)', alpha=0.7, s=20)
        if r_points:
            r_hrs, r_wrs = zip(*r_points)
            ax.scatter(r_hrs, r_wrs, c='blue', label='r (Run)', alpha=0.7, s=20)
        
        ax.set_xlabel('HP / Max HP')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'{model_name} (Show)\nTotal: {total_points}, C: {continue_ratio:.2f}, R: {run_ratio:.2f}')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # If no show file exists, create empty subplot
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 0.5, f'{model_name}\n(No Show Data)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name} (Show)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Store empty data
        stats_data['show']['continue_rate'].append(0)
        stats_data['show']['run_rate'].append(0)
        stats_data['show']['continue_high_wr'].append(0)
        stats_data['show']['run_low_wr'].append(0)

    # Second row: no show files
    if model_name in no_show_dict:
        filename = no_show_dict[model_name]
        txt_path = os.path.join(base_dir, filename)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        win_rates = []
        hp_ratios = []
        choices = []
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            m1 = re.search(r'Win Rate: ([\d\.]+), Choice: ([cr])', lines[i])
            m2 = re.search(r'HP ([\d\.]+)/([\d\.]+)', lines[i+1])
            if m1 and m2:
                win_rate = float(m1.group(1))
                choice = m1.group(2)
                hp = float(m2.group(1))
                max_hp = float(m2.group(2))
                hp_ratio = hp / max_hp if max_hp > 0 else 0
                win_rates.append(win_rate)
                hp_ratios.append(hp_ratio)
                choices.append(choice)

        # Calculate statistics
        total_points = len(choices)
        continue_count = choices.count('c')
        run_count = choices.count('r')
        continue_ratio = continue_count / total_points if total_points > 0 else 0
        run_ratio = run_count / total_points if total_points > 0 else 0
        
        # Continue rate when win rate > 0.5
        high_wr_choices = [ch for ch, wr in zip(choices, win_rates) if wr > 0.5]
        continue_high_wr_ratio = high_wr_choices.count('c') / len(high_wr_choices) if high_wr_choices else 0
        
        # Run rate when win rate < 0.5
        low_wr_choices = [ch for ch, wr in zip(choices, win_rates) if wr < 0.5]
        run_low_wr_ratio = low_wr_choices.count('r') / len(low_wr_choices) if low_wr_choices else 0
        
        # Store statistics data
        stats_data['no_show']['continue_rate'].append(continue_ratio)
        stats_data['no_show']['run_rate'].append(run_ratio)
        stats_data['no_show']['continue_high_wr'].append(continue_high_wr_ratio)
        stats_data['no_show']['run_low_wr'].append(run_low_wr_ratio)

        # Create subplot (second row)
        ax = fig.add_subplot(gs[1, col_idx])

        # Plot points
        c_points = [(hr, wr) for hr, wr, ch in zip(hp_ratios, win_rates, choices) if ch == 'c']
        r_points = [(hr, wr) for hr, wr, ch in zip(hp_ratios, win_rates, choices) if ch == 'r']
        
        if c_points:
            c_hrs, c_wrs = zip(*c_points)
            ax.scatter(c_hrs, c_wrs, c='red', label='c (Continue)', alpha=0.7, s=20)
        if r_points:
            r_hrs, r_wrs = zip(*r_points)
            ax.scatter(r_hrs, r_wrs, c='blue', label='r (Run)', alpha=0.7, s=20)
        
        ax.set_xlabel('HP / Max HP')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'{model_name} (No Show)\nTotal: {total_points}, C: {continue_ratio:.2f}, R: {run_ratio:.2f}')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # If no no_show file exists, create empty subplot
        ax = fig.add_subplot(gs[1, col_idx])
        ax.text(0.5, 0.5, f'{model_name}\n(No Data)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name} (No Show)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Store empty data
        stats_data['no_show']['continue_rate'].append(0)
        stats_data['no_show']['run_rate'].append(0)
        stats_data['no_show']['continue_high_wr'].append(0)
        stats_data['no_show']['run_low_wr'].append(0)

# Third row: statistics bar charts
x = np.arange(len(all_models))
width = 0.35

# 1. Total Continue Rate
ax1 = fig.add_subplot(gs[2, 0])
ax1.bar(x - width/2, stats_data['show']['continue_rate'], width, label='Show Battle Data', alpha=0.8)
ax1.bar(x + width/2, stats_data['no_show']['continue_rate'], width, label='Hide Battle Data', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('Continue Rate')
ax1.set_title('Total Continue Rate')
ax1.set_xticks(x)
ax1.set_xticklabels(all_models, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Total Run Rate
ax2 = fig.add_subplot(gs[2, 1])
ax2.bar(x - width/2, stats_data['show']['run_rate'], width, label='Show Battle Data', alpha=0.8)
ax2.bar(x + width/2, stats_data['no_show']['run_rate'], width, label='Hide Battle Data', alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('Run Rate')
ax2.set_title('Total Run Rate')
ax2.set_xticks(x)
ax2.set_xticklabels(all_models, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Continue Rate when Win Rate > 0.5
ax3 = fig.add_subplot(gs[2, 2])
ax3.bar(x - width/2, stats_data['show']['continue_high_wr'], width, label='Show Battle Data', alpha=0.8)
ax3.bar(x + width/2, stats_data['no_show']['continue_high_wr'], width, label='Hide Battle Data', alpha=0.8)
ax3.set_xlabel('Model')
ax3.set_ylabel('Continue Rate')
ax3.set_title('Continue Rate (Win Rate > 0.5)')
ax3.set_xticks(x)
ax3.set_xticklabels(all_models, rotation=45)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Run Rate when Win Rate < 0.5
ax4 = fig.add_subplot(gs[2, 3])
ax4.bar(x - width/2, stats_data['show']['run_low_wr'], width, label='Show Battle Data', alpha=0.8)
ax4.bar(x + width/2, stats_data['no_show']['run_low_wr'], width, label='Hide Battle Data', alpha=0.8)
ax4.set_xlabel('Model')
ax4.set_ylabel('Run Rate')
ax4.set_title('Run Rate (Win Rate < 0.5)')
ax4.set_xticks(x)
ax4.set_xticklabels(all_models, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save the combined image
output_path = os.path.join(base_dir, 'combined_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Combined image generated: {output_path}")
print(f"Number of models: {len(all_models)}")
print(f"Show files: {len(show_files)}")
print(f"No Show files: {len(no_show_files)}")