import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_model_name(filename):
    """Extract model name from filename"""
    # Remove .txt extension and show-related suffixes
    name = filename.replace('.txt', '').replace('_show', '').replace('show_', '')
    return name

def parse_txt_file(filepath):
    """Parse single txt file, extract difficulty, game index, win rate, and choice"""
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match format: Difficulty Easy, Game 1, Steps 7, Turn 5, Win Rate: 1.000, Choice: c
        match = re.search(r'Difficulty (\w+), Game (\d+),.*?Win Rate: ([\d\.]+), Choice: ([cr])', line)
        if match:
            difficulty = match.group(1)
            game_index = int(match.group(2))
            win_rate = float(match.group(3))
            choice = match.group(4)
            
            records.append({
                'difficulty': difficulty,
                'game_index': game_index,
                'win_rate': win_rate,
                'choice': choice
            })
    
    return records

def calculate_run_rates(records):
    """Calculate run rate (run selection rate when win rate < 0.5)"""
    # Group by difficulty and game index
    stats = {}
    
    for record in records:
        key = (record['difficulty'], record['game_index'])
        if record['win_rate'] < 0.5:  # Only consider cases where win rate < 0.5
            if key not in stats:
                stats[key] = []
            stats[key].append(record['choice'])
    
    # Calculate run rate
    run_rates = {}
    for (difficulty, game_index), choices in stats.items():
        if choices:  # Ensure there is data
            run_count = choices.count('r')
            run_rate = run_count / len(choices)
            run_rates[(difficulty, game_index)] = run_rate
    
    return run_rates

def create_run_rate_charts(logs_dir):
    """Create run rate line charts"""
    # Collect all txt files
    show_files = []
    no_show_files = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('.txt') and filename != 'plot.py':
            if 'show' in filename.lower():
                show_files.append(filename)
            else:
                no_show_files.append(filename)
    
    # Get all unique model names
    all_models = set()
    for filename in show_files + no_show_files:
        all_models.add(get_model_name(filename))
    all_models = sorted(list(all_models))
    
    if not all_models:
        print("No txt files found")
        return
    
    # Difficulty order
    difficulties = ['Easy', 'Normal', 'Hard', 'Insane']
    
    # Assign colors to each model
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#a6cee3', '#fb9a99', '#fdbea7', '#cab2d6', '#8dd3c7']
    
    # Ensure enough colors
    while len(color_list) < len(all_models):
        color_list.extend(color_list)
    
    model_colors = dict(zip(all_models, color_list[:len(all_models)]))
    
    # Create two subplots: show data and no show data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # First plot: Show battle data
    ax1.set_title('Show Battle Data', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Difficulty', fontsize=12)
    ax1.set_ylabel('Run Rate (Win Rate < 0.5)', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Second plot: No show battle data
    ax2.set_title('No Show Battle Data', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Difficulty', fontsize=12)
    ax2.set_ylabel('Run Rate (Win Rate < 0.5)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Process show files
    for model_name in all_models:
        show_file = None
        for filename in show_files:
            if get_model_name(filename) == model_name:
                show_file = filename
                break
        
        if show_file:
            filepath = os.path.join(logs_dir, show_file)
            records = parse_txt_file(filepath)
            run_rates = calculate_run_rates(records)
            
            color = model_colors[model_name]
            
            # Calculate run rate for each difficulty
            early_rates = []  # Game index <= 5
            late_rates = []   # Game index > 5
            
            for difficulty in difficulties:
                # Collect early games (game index <= 5)
                early_choices = []
                for (diff, game_idx), rate in run_rates.items():
                    if diff == difficulty and game_idx <= 5:
                        # Recalculate run rate for this group
                        for record in records:
                            if (record['difficulty'] == difficulty and 
                                record['game_index'] <= 5 and 
                                record['win_rate'] < 0.5):
                                early_choices.append(record['choice'])
                
                if early_choices:
                    early_run_rate = early_choices.count('r') / len(early_choices)
                    early_rates.append(early_run_rate)
                else:
                    early_rates.append(0)
                
                # Collect late games (game index > 5)
                late_choices = []
                for (diff, game_idx), rate in run_rates.items():
                    if diff == difficulty and game_idx > 5:
                        # Recalculate run rate for this group
                        for record in records:
                            if (record['difficulty'] == difficulty and 
                                record['game_index'] > 5 and 
                                record['win_rate'] < 0.5):
                                late_choices.append(record['choice'])
                
                if late_choices:
                    late_run_rate = late_choices.count('r') / len(late_choices)
                    late_rates.append(late_run_rate)
                else:
                    late_rates.append(0)
            
            # Draw early games line (game index <= 5)
            if any(rate > 0 for rate in early_rates):
                ax1.plot(difficulties, early_rates, 'o-', color=color, linewidth=2, 
                        markersize=6, label=f'{model_name} (≤5)')
            
            # Draw late games line (game index > 5)
            if any(rate > 0 for rate in late_rates):
                ax1.plot(difficulties, late_rates, 's--', color=color, linewidth=2, 
                        markersize=6, label=f'{model_name} (>5)')
    
    # Process no show files
    for model_name in all_models:
        no_show_file = None
        for filename in no_show_files:
            if get_model_name(filename) == model_name:
                no_show_file = filename
                break
        
        if no_show_file:
            filepath = os.path.join(logs_dir, no_show_file)
            records = parse_txt_file(filepath)
            run_rates = calculate_run_rates(records)
            
            color = model_colors[model_name]
            
            # Calculate run rate for each difficulty
            early_rates = []  # Game index <= 5
            late_rates = []   # Game index > 5
            
            for difficulty in difficulties:
                # Collect early games (game index <= 5)
                early_choices = []
                for (diff, game_idx), rate in run_rates.items():
                    if diff == difficulty and game_idx <= 5:
                        # Recalculate run rate for this group
                        for record in records:
                            if (record['difficulty'] == difficulty and 
                                record['game_index'] <= 5 and 
                                record['win_rate'] < 0.5):
                                early_choices.append(record['choice'])
                
                if early_choices:
                    early_run_rate = early_choices.count('r') / len(early_choices)
                    early_rates.append(early_run_rate)
                else:
                    early_rates.append(0)
                
                # Collect late games (game index > 5)
                late_choices = []
                for (diff, game_idx), rate in run_rates.items():
                    if diff == difficulty and game_idx > 5:
                        # Recalculate run rate for this group
                        for record in records:
                            if (record['difficulty'] == difficulty and 
                                record['game_index'] > 5 and 
                                record['win_rate'] < 0.5):
                                late_choices.append(record['choice'])
                
                if late_choices:
                    late_run_rate = late_choices.count('r') / len(late_choices)
                    late_rates.append(late_run_rate)
                else:
                    late_rates.append(0)
            
            # Draw early games line (game index <= 5)
            if any(rate > 0 for rate in early_rates):
                ax2.plot(difficulties, early_rates, 'o-', color=color, linewidth=2, 
                        markersize=6, label=f'{model_name} (≤5)')
            
            # Draw late games line (game index > 5)
            if any(rate > 0 for rate in late_rates):
                ax2.plot(difficulties, late_rates, 's--', color=color, linewidth=2, 
                        markersize=6, label=f'{model_name} (>5)')
    
    # Add legend
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set x-axis labels
    ax1.set_xticks(range(len(difficulties)))
    ax1.set_xticklabels(difficulties)
    ax2.set_xticks(range(len(difficulties)))
    ax2.set_xticklabels(difficulties)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig('run_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function"""
    logs_dir = 'logs'
    
    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} directory not found")
        return
    
    print("Analyzing run rate data...")
    fig = create_run_rate_charts(logs_dir)
    
    print("\nCharts saved as 'run_rate_analysis.png'")

if __name__ == "__main__":
    main() 