# Roguelike Game Analysis Project

A roguelike game project that analyzes different AI models' performance in a text-based dungeon crawler game.

## Features

- Text-based roguelike game with multiple difficulty levels
- Support for multiple AI models (OpenAI, Gemini, Ollama)
- Comprehensive game analysis and statistics
- Battle data visualization and analysis
- Performance metrics including win rate, average steps, and quit rate

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### API Keys Setup

Before running the game, you need to configure your API keys:

1. **OpenAI API Key**: 
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Set it in the `config.py` file or as an environment variable

2. **Gemini API Key**:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set it in the `config.py` file or as an environment variable

3. **Ollama**:
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Run locally on `http://localhost:11434`

### Configuration File

Create a `config.py` file in the project root:

```python
# API Configuration
OPENAI_API_KEY = "your_openai_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"

# Model Configuration
MODEL_NAME = "gpt-4o-mini"  # or your preferred model
```

## Usage

### Running the Game

```bash
python main.py
```

### Running Analysis

```bash
python analyze.py
```

## Project Structure

- `main.py` - Main game logic and AI model integration
- `analyze.py` - Data analysis and visualization
- `data.txt` - Game results data
- `logs/` - Detailed game logs for each model
- `*.png` - Generated analysis charts

## Game Modes

- **Easy**: Beginner-friendly gameplay
- **Normal**: Standard difficulty
- **Hard**: Challenging gameplay
- **Insane**: Extreme difficulty

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 