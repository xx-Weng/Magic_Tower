import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_data_file(filename):
    """Parse data file and extract game records"""
    records = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Find game ID and result
        if line and ':' in line and line.split(':')[0].isdigit():
            parts = line.split(':')
            game_id = int(parts[0])
            result = parts[1].strip()
            
            # Read subsequent information
            model = ""
            show_data = False
            difficulty = ""
            game_index = 0
            steps = 0
            final_floor = 0
            
            j = i + 1
            while j < len(lines) and (j == i + 1 or not (lines[j].strip() and lines[j].strip().split(':')[0].isdigit())):
                sub_line = lines[j].strip()
                if sub_line.startswith('Model:'):
                    model = sub_line.split('Model:')[1].strip()
                elif sub_line.startswith('Show battle data:'):
                    show_data = sub_line.split('Show battle data:')[1].strip() == 'True'
                elif sub_line.startswith('Difficulty:'):
                    difficulty = sub_line.split('Difficulty:')[1].strip()
                elif sub_line.startswith('Game index:'):
                    game_index = int(sub_line.split('Game index:')[1].strip())
                elif sub_line.startswith('Steps:'):
                    steps = int(sub_line.split('Steps:')[1].strip())
                elif sub_line.startswith('Final floor:'):
                    final_floor = int(sub_line.split('Final floor:')[1].strip())
                j += 1
            
            if model and difficulty:  # Ensure sufficient data
                records.append({
                    'game_id': game_id,
                    'result': result,
                    'model': model,
                    'show_data': show_data,
                    'difficulty': difficulty,
                    'game_index': game_index,
                    'steps': steps,
                    'final_floor': final_floor
                })
            
            i = j - 1
        i += 1
    
    return records

def calculate_win_rates(records):
    """Calculate win rates for each model under different conditions"""
    # Group statistics by model, show_data, difficulty, game_index
    stats = {}
    
    for record in records:
        key = (record['model'], record['show_data'], record['difficulty'])
        game_index = record['game_index']
        result = record['result']
        
        # Convert result to numeric value: Win=1, Die=0, Quit=0
        win_value = 1 if result == 'Win' else 0
        
        if key not in stats:
            stats[key] = {}
        if game_index not in stats[key]:
            stats[key][game_index] = []
        
        stats[key][game_index].append(win_value)
    
    # Calculate win rates
    win_rates = {}
    
    for (model, show_data, difficulty), game_indices in stats.items():
        # Calculate win rate for game_index <= 5
        early_games = []
        for idx in range(1, 6):
            if idx in game_indices:
                early_games.extend(game_indices[idx])
        
        # Calculate win rate for game_index >= 6
        late_games = []
        for idx in range(6, 11):
            if idx in game_indices:
                late_games.extend(game_indices[idx])
        
        # Only calculate win rate when there is sufficient data
        if early_games:
            early_win_rate = sum(early_games) / len(early_games) if early_games else 0
            win_rates[(model, show_data, difficulty, 'early')] = early_win_rate
        
        if late_games:
            late_win_rate = sum(late_games) / len(late_games) if late_games else 0
            win_rates[(model, show_data, difficulty, 'late')] = late_win_rate
    
    return win_rates

def create_win_rate_charts(win_rates):
    """Create win rate line charts"""
    # Difficulty order
    difficulties = ['Easy', 'Normal', 'Hard', 'Insane']
    
    # Get all models
    models = set()
    for key in win_rates.keys():
        models.add(key[0])
    
    # Assign colors to each model - use deeper colors to avoid yellow
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#a6cee3', '#fb9a99', '#fdbea7', '#cab2d6', '#ffff99']
    
    # Ensure sufficient colors
    while len(color_list) < len(models):
        color_list.extend(color_list)
    
    model_colors = dict(zip(sorted(models), color_list[:len(models)]))
    
    # Create two subplots: show data and no show data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # First chart: Show data = True
    ax1.set_title('Show Battle Data: True', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Difficulty', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Second chart: Show data = False
    ax2.set_title('Show Battle Data: False', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Difficulty', fontsize=12)
    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Draw lines for each model
    for model in sorted(models):
        color = model_colors[model]
        
        # Show data = True
        early_rates = []
        late_rates = []
        
        for difficulty in difficulties:
            early_key = (model, True, difficulty, 'early')
            late_key = (model, True, difficulty, 'late')
            
            early_rate = win_rates.get(early_key, 0)
            late_rate = win_rates.get(late_key, 0)
            
            early_rates.append(early_rate)
            late_rates.append(late_rate)
        
        # Draw early games line (game index <= 5)
        if any(rate > 0 for rate in early_rates):
            ax1.plot(difficulties, early_rates, 'o-', color=color, linewidth=2, 
                    markersize=6, label=f'{model} (≤5)')
        
        # Draw late games line (game index >= 6)
        if any(rate > 0 for rate in late_rates):
            ax1.plot(difficulties, late_rates, 's--', color=color, linewidth=2, 
                    markersize=6, label=f'{model} (≥6)')
        
        # Show data = False
        early_rates = []
        late_rates = []
        
        for difficulty in difficulties:
            early_key = (model, False, difficulty, 'early')
            late_key = (model, False, difficulty, 'late')
            
            early_rate = win_rates.get(early_key, 0)
            late_rate = win_rates.get(late_key, 0)
            
            early_rates.append(early_rate)
            late_rates.append(late_rate)
        
        # Draw early games line (game index <= 5)
        if any(rate > 0 for rate in early_rates):
            ax2.plot(difficulties, early_rates, 'o-', color=color, linewidth=2, 
                    markersize=6, label=f'{model} (≤5)')
        
        # Draw late games line (game index >= 6)
        if any(rate > 0 for rate in late_rates):
            ax2.plot(difficulties, late_rates, 's--', color=color, linewidth=2, 
                    markersize=6, label=f'{model} (≥6)')
    
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
    plt.savefig('win_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_summary_stats(win_rates):
    """Print statistics summary"""
    print("Win Rate Statistics Summary:")
    print("=" * 80)
    
    # Group by model
    models = set()
    for key in win_rates.keys():
        models.add(key[0])
    
    for model in sorted(models):
        print(f"\nModel: {model}")
        print("-" * 40)
        
        for show_data in [True, False]:
            print(f"Show battle data: {show_data}")
            for difficulty in ['Easy', 'Normal', 'Hard', 'Insane']:
                early_key = (model, show_data, difficulty, 'early')
                late_key = (model, show_data, difficulty, 'late')
                
                early_rate = win_rates.get(early_key, 0)
                late_rate = win_rates.get(late_key, 0)
                
                if early_rate > 0 or late_rate > 0:
                    print(f"  {difficulty}: Early (≤5)={early_rate:.3f}, Late (≥6)={late_rate:.3f}")

def main():
    """Main function"""
    print("Parsing data file...")
    records = parse_data_file('data.txt')
    print(f"Parsed {len(records)} game records")
    
    print("\nCalculating win rates...")
    win_rates = calculate_win_rates(records)
    
    print("\nGenerating charts...")
    fig = create_win_rate_charts(win_rates)
    
    print("\nCharts saved as 'win_rate_analysis.png'")
    
    # Print summary statistics
    print_summary_stats(win_rates)

if __name__ == "__main__":
    main() 