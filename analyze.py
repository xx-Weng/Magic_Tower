import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# 设置字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_data_file(filename):
    """Parse data file"""
    data = []
    current_record = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_record:
                    data.append(current_record)
                    current_record = {}
                continue
            
            # Parse game result
            if re.match(r'^\d+:', line):
                result = line.split(':')[1].strip()
                current_record['result'] = result
            elif line.startswith('Model:'):
                current_record['model'] = line.split(':', 1)[1].strip()
            elif line.startswith('Show battle data:'):
                current_record['show_battle_data'] = line.split(':', 1)[1].strip() == 'True'
            elif line.startswith('Difficulty:'):
                current_record['difficulty'] = line.split(':', 1)[1].strip()
            elif line.startswith('Steps:'):
                current_record['steps'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('Final floor:'):
                current_record['final_floor'] = int(line.split(':', 1)[1].strip())
    
    # Add the last record
    if current_record:
        data.append(current_record)
    
    return pd.DataFrame(data)

def calculate_win_rate(df, model, difficulty, show_battle_data):
    """Calculate win rate"""
    subset = df[(df['model'] == model) & 
                (df['difficulty'] == difficulty) & 
                (df['show_battle_data'] == show_battle_data)]
    
    if len(subset) == 0:
        return 0
    
    wins = len(subset[subset['result'] == 'Win'])
    return wins / len(subset) * 100

def calculate_avg_final_floor(df, model, difficulty, show_battle_data):
    """Calculate average final floor"""
    subset = df[(df['model'] == model) & 
                (df['difficulty'] == difficulty) & 
                (df['show_battle_data'] == show_battle_data)]
    
    if len(subset) == 0:
        return 0
    
    return subset['final_floor'].mean()

def calculate_avg_steps_on_win(df, model, difficulty, show_battle_data):
    """Calculate average steps for wins only"""
    subset = df[(df['model'] == model) & \
                (df['difficulty'] == difficulty) & \
                (df['show_battle_data'] == show_battle_data) & \
                (df['result'] == 'Win')]
    if len(subset) == 0:
        return 0
    return subset['steps'].mean()

def calculate_quit_rate(df, model, difficulty, show_battle_data):
    """Calculate quit rate (%)"""
    subset = df[(df['model'] == model) & \
                (df['difficulty'] == difficulty) & \
                (df['show_battle_data'] == show_battle_data)]
    if len(subset) == 0:
        return 0
    quits = len(subset[subset['result'] == 'Quit'])
    return quits / len(subset) * 100

def create_bar_charts(df):
    """Create bar charts: 4 rows (difficulty), 4 columns (win rate, avg final floor, avg steps on win, quit rate)"""
    models = df['model'].unique()
    difficulties = ['Easy', 'Normal', 'Hard', 'Insane']
    titles = [
        'Easy Mode Win Rate', 'Easy Mode Average Final Floor', 'Easy Mode Avg Steps (Win)', 'Easy Mode Quit Rate',
        'Normal Mode Win Rate', 'Normal Mode Average Final Floor', 'Normal Mode Avg Steps (Win)', 'Normal Mode Quit Rate',
        'Hard Mode Win Rate', 'Hard Mode Average Final Floor', 'Hard Mode Avg Steps (Win)', 'Hard Mode Quit Rate',
        'Insane Mode Win Rate', 'Insane Mode Average Final Floor', 'Insane Mode Avg Steps (Win)', 'Insane Mode Quit Rate',
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(36, 22))
    fig.suptitle('Game Data Analysis', fontsize=18, fontweight='bold')
    
    width = 0.35
    x = np.arange(len(models))
    
    for i, difficulty in enumerate(difficulties):
        # Win Rate
        win_rates_show = [calculate_win_rate(df, model, difficulty, True) for model in models]
        win_rates_no_show = [calculate_win_rate(df, model, difficulty, False) for model in models]
        ax_win = axes[i, 0]
        ax_win.bar(x - width/2, win_rates_show, width, label='Show Battle Data', alpha=0.8)
        ax_win.bar(x + width/2, win_rates_no_show, width, label='Hide Battle Data', alpha=0.8)
        ax_win.set_xlabel('Model')
        ax_win.set_ylabel('Win Rate (%)')
        ax_win.set_title(f'{difficulty} Mode Win Rate')
        ax_win.set_xticks(x)
        ax_win.set_xticklabels(models, rotation=45)
        ax_win.legend()
        ax_win.grid(True, alpha=0.3)
        ax_win.set_ylim(0, 110)
        
        # Average Final Floor
        avg_floors_show = [calculate_avg_final_floor(df, model, difficulty, True) for model in models]
        avg_floors_no_show = [calculate_avg_final_floor(df, model, difficulty, False) for model in models]
        ax_floor = axes[i, 1]
        ax_floor.bar(x - width/2, avg_floors_show, width, label='Show Battle Data', alpha=0.8)
        ax_floor.bar(x + width/2, avg_floors_no_show, width, label='Hide Battle Data', alpha=0.8)
        ax_floor.set_xlabel('Model')
        ax_floor.set_ylabel('Average Final Floor')
        ax_floor.set_title(f'{difficulty} Mode Average Final Floor')
        ax_floor.set_xticks(x)
        ax_floor.set_xticklabels(models, rotation=45)
        ax_floor.legend()
        ax_floor.grid(True, alpha=0.3)
        ax_floor.set_ylim(0, 3.4)
        
        # Average Steps (Win Only)
        avg_steps_show = [calculate_avg_steps_on_win(df, model, difficulty, True) for model in models]
        avg_steps_no_show = [calculate_avg_steps_on_win(df, model, difficulty, False) for model in models]
        ax_steps = axes[i, 2]
        ax_steps.bar(x - width/2, avg_steps_show, width, label='Show Battle Data', alpha=0.8)
        ax_steps.bar(x + width/2, avg_steps_no_show, width, label='Hide Battle Data', alpha=0.8)
        ax_steps.set_xlabel('Model')
        ax_steps.set_ylabel('Average Steps (Win Only)')
        ax_steps.set_title(f'{difficulty} Mode Avg Steps (Win)')
        ax_steps.set_xticks(x)
        ax_steps.set_xticklabels(models, rotation=45)
        ax_steps.legend()
        ax_steps.grid(True, alpha=0.3)
        ax_steps.set_ylim(0, 55)
        
        # Quit Rate
        quit_rates_show = [calculate_quit_rate(df, model, difficulty, True) for model in models]
        quit_rates_no_show = [calculate_quit_rate(df, model, difficulty, False) for model in models]
        ax_quit = axes[i, 3]
        ax_quit.bar(x - width/2, quit_rates_show, width, label='Show Battle Data', alpha=0.8)
        ax_quit.bar(x + width/2, quit_rates_no_show, width, label='Hide Battle Data', alpha=0.8)
        ax_quit.set_xlabel('Model')
        ax_quit.set_ylabel('Quit Rate (%)')
        ax_quit.set_title(f'{difficulty} Mode Quit Rate')
        ax_quit.set_xticks(x)
        ax_quit.set_xticklabels(models, rotation=45)
        ax_quit.legend()
        ax_quit.grid(True, alpha=0.3)
        ax_quit.set_ylim(0, 110)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('game_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(df):
    """Print statistics"""
    print("=== Data Statistics ===")
    print(f"Total records: {len(df)}")
    print(f"Number of models: {len(df['model'].unique())}")
    print(f"Model list: {list(df['model'].unique())}")
    print(f"Difficulty levels: {list(df['difficulty'].unique())}")
    print(f"Game result distribution:")
    print(df['result'].value_counts())
    print("\n=== Model Statistics ===")
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\nModel: {model}")
        print(f"  Total games: {len(model_data)}")
        print(f"  Win rate: {len(model_data[model_data['result'] == 'Win']) / len(model_data) * 100:.1f}%")
        print(f"  Average final floor: {model_data['final_floor'].mean():.2f}")
    
    print("\n=== Detailed Win Rate Analysis ===")
    difficulties = ['Easy', 'Normal', 'Hard', 'Insane']
    
    for model in df['model'].unique():
        print(f"\n{model}:")
        for difficulty in difficulties:
            for show_battle_data in [True, False]:
                win_rate = calculate_win_rate(df, model, difficulty, show_battle_data)
                battle_data_str = "Show Battle Data" if show_battle_data else "Hide Battle Data"
                print(f"  {difficulty} Mode ({battle_data_str}): {win_rate:.1f}%")

def main():
    # Parse data
    df = parse_data_file('data.txt')
    
    # Print statistics
    print_statistics(df)
    
    # Create bar charts
    create_bar_charts(df)
    
    print("\nChart saved as 'game_analysis.png'")

if __name__ == "__main__":
    main() 