import requests
import time 
import os   
import random 
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Tuple
from math import ceil
import re
from openai import OpenAI

# Import configuration
try:
    from config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_NAME, MAX_FLOOR, MAX_GAMES
except ImportError:
    # Fallback configuration - users should create config.py with their API keys
    print("Warning: config.py not found. Please create config.py with your API keys.")
    print("See README.md for configuration instructions.")
    
    # Default values - users must set these in config.py
    OPENAI_API_KEY = None
    GEMINI_API_KEY = None
    MODEL_NAME = 'gpt-4o-mini'  # Default model name
    MAX_FLOOR = 3  # Maximum number of floors, used for boss levels and victory conditions
    MAX_GAMES = 10  # Maximum number of games to play


class OllamaChat:
    def __init__(self, model_name, base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.messages = [] # Stores the conversation history for the session

    def send_message(self, current_prompt_content, instruction_tip=None):
        """
        Sends the current prompt to the Ollama API and attempts to extract
        the last character (or last single-character word) as the action.
        The full prompt (current_prompt_content + instruction_tip) is added as a user message.
        The raw assistant response is added to history.
        """
        full_user_content = current_prompt_content
        if instruction_tip:
            full_user_content += f"\n({instruction_tip})"

        # Add the current user's full context/prompt for this turn to the history
        # that will be sent to the API
        messages_to_send = self.messages + [{
            "role": "user",
            "content": full_user_content
        }]

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages_to_send, # Send history + current user message
            "stream": False
        }

        try:
            
            response = requests.post(url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            result = response.json()
            raw_assistant_message = result['message']['content'] # The full, raw response from LLM

            print(f"LLM Raw Response: {raw_assistant_message}") # Log the raw response

            # Update the persistent history with what was actually sent and received
            self.messages.append({"role": "user", "content": full_user_content}) # Add the user message that led to this response
            self.messages.append({"role": "assistant", "content": raw_assistant_message}) # Add the raw assistant response

            # Process the raw_assistant_message to extract the action (last character/word)
            if raw_assistant_message:
                cleaned_response = raw_assistant_message.strip().lower()
                words = cleaned_response.split()
                if words:
                    last_word = words[-1]
                    # Check if the last word is a single character (common for actions like w,a,s,d,1,r)
                    # You might want to expand the list of valid single character actions
                    if len(last_word) == 1 and last_word in "wasdgiqr012345":
                        return last_word
                    # If not a single char action, maybe it's a multi-char "Error." or similar
                    # For now, let's try just returning the last character of the last word if it's not a clear action
                    # This is a bit of a heuristic.
                    # A more robust way would be to check if the last_word is a known valid action string.
                    # Or, as a simpler heuristic, just take the very last character of the entire cleaned response.
                    if cleaned_response: # Ensure cleaned_response is not empty
                        return cleaned_response[-1] # Return the absolute last character
                elif cleaned_response: # If no words but cleaned_response is not empty (e.g. "w")
                    return cleaned_response[0] # Return the first (and only) character
            
            # If no suitable action character found after processing
            print("Warning: Could not extract a clear single-character action from LLM response.")
            return " " # Return a space or other indicator of "no valid action extracted"

        except requests.exceptions.RequestException as e:
            print(f"Ollama API error: {e}")
            return "Error." # Return a generic error message
        except Exception as e_json: # Catch potential JSON parsing errors or KeyError
            print(f"Error processing LLM response: {e_json}")
            print(f"Raw response data: {response.text if 'response' in locals() else 'N/A'}")
            return "Error."


    def clear_history(self):
        """Clear the chat history, including user and assistant messages."""
        self.messages = []
        #print("DEBUG: Chat history cleared.")

class GeminiChat:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.messages = []

    def send_message(self, current_prompt_content, instruction_tip=None):
        import requests
        import time
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        prompt = current_prompt_content
        if instruction_tip:
            prompt += f"\n({instruction_tip})"
        # Construct message history
        contents = self.messages.copy()
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        data = {"contents": contents}
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            raw_assistant_message = result['candidates'][0]['content']['parts'][0]['text']
            print(f"Gemini Raw Response: {raw_assistant_message}")
            # Append current conversation to history
            self.messages.append({"role": "user", "parts": [{"text": prompt}]})
            self.messages.append({"role": "model", "parts": [{"text": raw_assistant_message}]})
            cleaned_response = raw_assistant_message.strip().lower()
            words = cleaned_response.split()
            if words:
                last_word = words[-1]
                if len(last_word) == 1 and last_word in "wasdgiqr012345":
                    #time.sleep(3)
                    return last_word
                if cleaned_response:
                    #time.sleep(3)
                    return cleaned_response[-1]
            elif cleaned_response:
                #time.sleep(3)
                return cleaned_response[0]
            print("Warning: Could not extract a clear single-character action from Gemini response.")
            #time.sleep(3)
            return " "
        except Exception as e:
            print(f"Gemini API error: {e}")
            #time.sleep(3)
            return "Error."

    def clear_history(self):
        self.messages = []

class OpenaiChat:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.messages = []  # Stores the conversation history for the session

    def send_message(self, current_prompt_content, instruction_tip=None):
        """
        Sends the current prompt to the OpenAI API and returns the model's response.
        The full prompt (current_prompt_content + instruction_tip) is added as a user message.
        The assistant response is added to history.
        """
        # Compose input as a list of messages (history + current)
        input_messages = self.messages.copy()
        user_content = current_prompt_content
        if instruction_tip:
            user_content += f"\n({instruction_tip})"
        input_messages.append({"role": "user", "content": user_content})
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=input_messages
            )
            reply = response.output_text
            print(f"OpenAI Raw Response: {reply}")  # Automatically output to console
            # Append to history
            self.messages.append({"role": "user", "content": user_content})
            self.messages.append({"role": "assistant", "content": reply})
            # Try to extract a single-character action (like OllamaChat)
            cleaned_response = reply.strip().lower()
            words = cleaned_response.split()
            if words:
                last_word = words[-1]
                if len(last_word) == 1 and last_word in "wasdgiqr012345":
                    return last_word
                if cleaned_response:
                    return cleaned_response[-1]
            elif cleaned_response:
                return cleaned_response[0]
            print("Warning: Could not extract a clear single-character action from OpenAI response.")
            return " "
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Error."

    def clear_history(self):
        self.messages = []

class GameOverException(Exception):
    pass

class Player:
    def __init__(self, difficulty='easy'):
        self.name = "Player"
        self.level = 1
        self.exp = 0
        self.exp_to_next_level = 1000
        self.hp = 100
        self.max_hp = 100
        self.attack = 10
        self.defense = 5
        self.attack_speed = 1.0
        self.gold = 50 if difficulty in ['hard', 'insane'] else 0
        self.items = []
        # Add life steal skill toggle
        self.life_steal_enabled = False  # Set to False to disable life steal
        # Add berserker skill toggle
        self.berserker_enabled = False  # Set to False to disable berserker
        self.skills = {
            'berserker': {
                'name': 'Berserker',
                'type': 'passive',
                'condition': lambda self: self.hp < self.max_hp * 0.5,
                'effect': lambda self: self.attack * 2
            },
            'life_steal': {
                'name': 'Life Steal',
                'type': 'passive',
                'condition': lambda self: True,
                'effect': lambda damage: damage * 0.2
            }
        }

    def gain_exp(self, amount):
        self.exp += amount
        while self.exp >= self.exp_to_next_level:
            self.level_up()

    def gain_gold(self, amount):
        self.gold += amount

    def level_up(self):
        global prompt_accumulator
        self.level += 1
        self.exp -= self.exp_to_next_level
        prompt_accumulator += f"\nLeveled up! Current level: {self.level}"
        options = self.generate_level_up_options()
        self.display_level_up_options(options)
        choice = self.get_level_up_choice() # This will call the LLM
        self.apply_level_up_choice(choice, options)

    def generate_level_up_options(self):
        options = []
        for _ in range(3):
            stat_type = random.choice(['hp', 'attack', 'defense', 'attack_speed'])
            boost_value = random.uniform(0.1, 0.3)
            options.append({
                'type': stat_type,
                'value': boost_value,
                'description': f'Increase {self.get_stat_name(stat_type)} by {int(boost_value*100)}%'
            })
        return options

    def display_level_up_options(self, options):
        global prompt_accumulator
        prompt_accumulator += "\nPlease choose your level up bonus:"
        for i, option in enumerate(options, 1):
            prompt_accumulator += f"\n{i}. {option['description']}"

    def get_level_up_choice(self):
        global prompt_accumulator, chat_session
        # This function assumes prompt_accumulator has already been populated
        # with "Leveled up! ..." and the options by the calling level_up() method.

        while True:
            # Add the specific question for the LLM to the existing accumulated prompt
            current_turn_context = prompt_accumulator + "\nPlease choose (1-3):"
            instruction = "Your response MUST be a single digit: 1, 2, or 3. No other text."

            print(current_turn_context) # Show the full context to the human user/developer
            
            # The OllamaChat class will append current_turn_context to its internal history
            # and then send the whole history.
            # If you want this call to be "stateless" regarding the chat_session's history
            # (i.e., only current_turn_context matters for this specific decision, and the LLM
            # shouldn't see previous turns' chat for *this specific choice*),
            # you would call chat_session.clear_history() before send_message.
            # However, the current OllamaChat is designed to maintain history.
            # The key is that `current_turn_context` (built from `prompt_accumulator`)
            # should contain *all information needed for this turn's decision*.
            
            # chat_session.clear_history() # Uncomment if you want each LLM call to be "fresh"
                                          # using only the user_prompt_content passed below,
                                          # effectively ignoring OllamaChat's internal history for this call.
                                          # This would make it behave more like your original Gemini setup
                                          # where `prompt` was the sole source of context for each call.

            extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction)
            
            # Clear prompt_accumulator after the LLM call for this specific interaction,
            # so it's fresh for the next game state display or interaction.
            prompt_accumulator = ""

            # The 'LLM Raw Response' is now printed inside chat_session.send_message
            # print("LLM Processed Action: ", extracted_action) # Already done by send_message or can be done here.


            if extracted_action == "Error.":
                prompt_accumulator += "LLM API error. Please enter a valid number!"
                # Potentially add a manual input fallback here if desired
                continue # Retry or handle error

            try:
                choice = int(extracted_action) # extracted_action should be '1', '2', or '3'
                if 1 <= choice <= 3:
                    return choice - 1 # Convert to 0-based index
                else:
                    prompt_accumulator += f"Invalid choice '{extracted_action}'. "
            except ValueError:
                prompt_accumulator += f"LLM response '{extracted_action}' was not a valid number. "
            
            # If loop continues, it means an invalid choice was made or parsed
            prompt_accumulator += "Please ensure the response is a single digit 1, 2, or 3."


    def apply_level_up_choice(self, choice, options):
        option = options[choice]
        if option['type'] == 'hp':
            self.max_hp = int(self.max_hp * (1 + option['value']))
            self.hp = self.max_hp
        elif option['type'] == 'attack':
            self.attack = round(self.attack * (1 + option['value']), 1)
        elif option['type'] == 'defense':
            self.defense = round(self.defense * (1 + option['value']), 1)
        elif option['type'] == 'attack_speed':
            self.attack_speed *= (1 + option['value'])

    def get_stat_name(self, stat):
        stat_names = {
            'hp': 'Max HP',
            'attack': 'Attack',
            'defense': 'Defense',
            'attack_speed': 'Attack Speed'
        }
        return stat_names.get(stat, stat)

    def use_item(self, item_index):
        global prompt_accumulator
        # This function assumes prompt_accumulator has been populated with item list
        # and the choice has been made via LLM through handle_item_use in Game class.
        item = self.items[item_index]
        if item['type'] == 'health_potion':
            heal_amount = int(self.max_hp * 0.5)
            self.hp = min(self.max_hp, self.hp + heal_amount)
            prompt_accumulator += f"Used Health Potion, restored {heal_amount} HP! (50% of max HP)"
        self.items.pop(item_index) # Remove item after use

    def take_damage(self, damage):
        self.hp = max(0, self.hp - damage)
        return damage

class Player_Human(Player):
    # override the different function between AI player and Human player
    def level_up(self):
        self.level += 1
        self.exp -= self.exp_to_next_level
        print(f"\nLevel Up! Current Level: {self.level}")
        options = self.generate_level_up_options()
        self.display_level_up_options(options)
        choice = self.get_level_up_choice()
        self.apply_level_up_choice(choice, options)

    def display_level_up_options(self, options):
        print("\nPlease choose your level up bonus:") 
        for i, option in enumerate(options, 1):
            print(f"{i}. {option['description']}")

    def get_level_up_choice(self):
        while True:
            try:
                choice = int(input("\nPlease choose (1-3): "))
                if 1 <= choice <= 3:
                    return choice - 1
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")

    def use_item(self, item_index):
        item = self.items[item_index]
        if item['type'] == 'health_potion':
            heal_amount = int(self.max_hp * 0.5)
            self.hp = min(self.max_hp, self.hp + heal_amount)
            print(f"Used Health Potion, recovered {heal_amount} HP! (50% of max HP)")
        self.items.pop(item_index)

class Monster:
    def __init__(self, floor, difficulty='easy', hp_factor=None, attack_factor=None, defense_factor=None, speed_factor=None):
        self.floor = floor
        self.base_hp = 30 # 50 originally
        self.base_attack = 3 # 5 originally
        self.base_defense = 1 # 2 originally
        self.base_attack_speed = 0.8
        self.gold_value = 20
        self.exp_value = 1000

        difficulty_multiplier = {
            'easy': 1.0,
            'normal': 1.5,
            'hard': 2.0,
            'insane': 2.0
        }.get(difficulty, 1.0)

        if difficulty == 'insane':
            power_multiplier = difficulty_multiplier * (1.3 ** (floor - 1)) # 2** originally
        else:
            power_multiplier = difficulty_multiplier * (1 + (floor - 1) * 0.3)

        # random factor for each attribute, 0.5-1.5
        hp_factor = random.uniform(0.5, 1.5)
        attack_factor = random.uniform(0.5, 1.5)
        defense_factor = random.uniform(0.5, 1.5)
        speed_factor = random.uniform(0.5, 1.5)

        self.hp = int(self.base_hp * power_multiplier * hp_factor)
        self.max_hp = self.hp
        self.attack = int(self.base_attack * power_multiplier * attack_factor)
        self.defense = int(self.base_defense * power_multiplier * defense_factor)
        self.attack_speed = round(self.base_attack_speed * power_multiplier * speed_factor, 2)
        self.name = "Monster"

    def take_damage(self, damage):
        self.hp = max(0, self.hp - damage)
        return damage

class Boss(Monster):
    def __init__(self, floor, difficulty='easy'):
        super().__init__(floor, difficulty)
        self.hp *= 2
        self.max_hp = self.hp
        self.attack *= 2
        self.defense *= 2
        self.attack_speed *= 2
        self.gold_value = 50
        self.name = "Boss"
        self.invincible_turns = 0
        self.invincible_threshold = 0.20
        self.invincible_triggered = False

    def take_damage(self, damage):
        global prompt_accumulator
        if self.hp <= self.max_hp * self.invincible_threshold and not self.invincible_triggered:
            self.invincible_turns = 5
            self.invincible_triggered = True
            prompt_accumulator += "\nBoss becomes invincible for 5 turns!"
            return 0
        elif self.invincible_turns > 0:
            prompt_accumulator += "\nBoss is invincible and takes no damage!"
            self.invincible_turns -= 1
            return 0
        return super().take_damage(damage)

class Boss_Human(Monster):
    def __init__(self, floor, difficulty='easy'):
        super().__init__(floor, difficulty)
        self.hp *= 2
        self.max_hp = self.hp
        self.attack *= 2
        self.defense *= 2
        self.attack_speed *= 2
        self.gold_value = 50
        self.name = "Boss"
        self.invincible_turns = 0
        self.invincible_threshold = 0.20
        self.invincible_triggered = False

    def take_damage(self, damage):
        if self.hp <= self.max_hp * self.invincible_threshold and not self.invincible_triggered:
            self.invincible_turns = 5
            self.invincible_triggered = True
            print("Boss becomes invincible for 5 turns!")
            return 0
        elif self.invincible_turns > 0:
            print("Boss is invincible and takes no damage!")
            self.invincible_turns -= 1
            return 0
        return super().take_damage(damage)   

class Map:
    def __init__(self, floor):
        self.floor = floor
        self.size = 4 if floor % MAX_FLOOR != 0 else 9
        self.grid = [['X' for _ in range(self.size)] for _ in range(self.size)]
        self.visible_cells = set()
        self.generate_map()

    def generate_map(self):
        if self.floor % MAX_FLOOR == 0:
            self.size = 9 
            self.grid = [['O' for _ in range(self.size)]] 
            self.entry_pos = (0, 0)
            self.grid[0][0] = 'Y' 
            self.grid[0][4] = 'M' 
            self.grid[0][8] = 'E' 
        else:
            self.size = 4 
            self.grid = [['O' for _ in range(self.size)] for _ in range(self.size)]
            self.entry_pos = (0, 0)
            self.grid[0][0] = 'Y' 
            self.grid[self.size-1][self.size-1] = 'E' 

            shop_pos = self.get_random_empty_position()
            self.grid[shop_pos[0]][shop_pos[1]] = 'S'

            event_pos = self.get_random_empty_position()
            self.grid[event_pos[0]][event_pos[1]] = '?'

            for _ in range(4):  # number of monsters each floor
                monster_pos = self.get_random_empty_position()
                self.grid[monster_pos[0]][monster_pos[1]] = 'M'

    def get_random_empty_position(self):
        while True:
            if self.floor % MAX_FLOOR == 0: 
                pos = (0, random.randint(0, self.size - 1))
                if 0 <= pos[0] < len(self.grid) and 0 <= pos[1] < len(self.grid[pos[0]]) and self.grid[pos[0]][pos[1]] == 'O':
                    return pos
            else: 
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if 0 <= pos[0] < len(self.grid) and 0 <= pos[1] < len(self.grid[pos[0]]) and self.grid[pos[0]][pos[1]] == 'O':
                    return pos


    def display_map(self, player_pos, vision_range):
        global prompt_accumulator
        prompt_accumulator += f"\nCurrent Floor: {self.floor}"
        prompt_accumulator += "\nMap:\n"
        map_height = 1 if self.floor % MAX_FLOOR == 0 else self.size
        for i in range(map_height):
            row_display = []
            for j in range(self.size):
                char_to_add = 'X' 
                if vision_range == self.size: 
                    cell = self.grid[i][j]
                    char_to_add = 'O' if cell == 'X' else cell 
                elif (i, j) in self.visible_cells:
                    cell = self.grid[i][j]
                    char_to_add = 'O' if cell == 'X' else cell
                row_display.append(char_to_add)
            prompt_accumulator += ' '.join(row_display) + '\n'

    def update_visibility(self, player_pos, vision_range):
        if self.floor % MAX_FLOOR == 0: 
            for j in range(max(0, player_pos[1] - vision_range),
                          min(self.size, player_pos[1] + vision_range + 1)):
                self.visible_cells.add((0, j))
        else: 
            for i in range(max(0, player_pos[0] - vision_range),
                          min(self.size, player_pos[0] + vision_range + 1)):
                for j in range(max(0, player_pos[1] - vision_range),
                              min(self.size, player_pos[1] + vision_range + 1)):
                    self.visible_cells.add((i, j))

class Map_Human(Map):
    def display_map(self, player_pos, vision_range):
        print(f"Current Floor: {self.floor}")
        print("Map:")
        map_height = 1 if self.floor % MAX_FLOOR == 0 else self.size
        for i in range(map_height):
            row_display = []
            for j in range(self.size):
                char_to_add = 'X' 
                if vision_range == self.size: 
                    cell = self.grid[i][j]
                    char_to_add = 'O' if cell == 'X' else cell 
                elif (i, j) in self.visible_cells:
                    cell = self.grid[i][j]
                    char_to_add = 'O' if cell == 'X' else cell
                row_display.append(char_to_add)
            print(' '.join(row_display))

class Shop:
    def __init__(self):
        self.slots = 5
        self.items = []
        self.refresh_cost = 10
        self.refresh_shop()

    def refresh_shop(self):
        self.items = []
        for _ in range(self.slots):
            item_type = random.choice(['health_potion', 'stat_boost'])
            if item_type == 'health_potion':
                self.items.append({
                    'type': 'health_potion',
                    'name': 'Health Potion',
                    'price': 10,
                    'effect': 'Restores 50% of max HP',
                    'sold': False
                })
            else:
                stat_type = random.choice(['hp', 'attack', 'attack_speed'])
                boost_value = random.uniform(0.1, 0.3)
                self.items.append({
                    'type': 'stat_boost',
                    'name': f'{self.get_stat_name(stat_type)} Boost',
                    'price': 10,
                    'stat': stat_type,
                    'value': boost_value,
                    'effect': f'Permanently increases {self.get_stat_name(stat_type)} by {int(boost_value*100)}%',
                    'sold': False
                })

    def get_stat_name(self, stat):
        stat_names = {
            'hp': 'Max HP',
            'attack': 'Attack',
            'attack_speed': 'Attack Speed'
        }
        return stat_names.get(stat, stat)

    def display_shop(self):
        global prompt_accumulator
        prompt_accumulator += "\nShop Items:"
        for i, item in enumerate(self.items, 1):
            if not item['sold']:
                prompt_accumulator += f"\n{i}. {item['name']} - {item['price']} Gold"
                prompt_accumulator += f"\n   Effect: {item['effect']}"
            else:
                prompt_accumulator += f"\n{i}. Sold Out"
        prompt_accumulator += f"\n\nEnter 'r' to refresh shop (costs {self.refresh_cost} Gold)"
        prompt_accumulator += "\nEnter '0' to exit shop"

    def buy_item(self, player, item_index):
        global prompt_accumulator
        if 0 <= item_index < len(self.items):
            item = self.items[item_index]
            if not item['sold'] and player.gold >= item['price']:
                player.gold -= item['price']
                item['sold'] = True
                self.apply_item_effect(player, item)
                prompt_accumulator += f"Purchase successful! {item['effect']}"
                return True
            elif item['sold']:
                prompt_accumulator += "This item is sold out!"
            else:
                prompt_accumulator += "Not enough gold!"
        else:
            prompt_accumulator += "Invalid item number!"
        return False

    def apply_item_effect(self, player, item):
        if item['type'] == 'health_potion':
            player.items.append({
                'type': 'health_potion',
                'name': 'Health Potion',
                'description': 'Restores 50% of max HP'
            })
        elif item['type'] == 'stat_boost':
            if item['stat'] == 'hp':
                player.max_hp = int(player.max_hp * (1 + item['value']))
                player.hp = player.max_hp
            elif item['stat'] == 'attack':
                player.attack = int(player.attack * (1 + item['value']))
            elif item['stat'] == 'attack_speed':
                player.attack_speed *= (1 + item['value'])

class Shop_Human(Shop):
    def display_shop(self):
        print("\nShop Items:")
        for i, item in enumerate(self.items, 1):
            if not item['sold']:
                print(f"\n{i}. {item['name']} - {item['price']} Gold") 
                print(f"   Effect: {item['effect']}")
            else:
                print(f"\n{i}. Sold Out")
        print(f"\nEnter 'r' to refresh shop (costs {self.refresh_cost} Gold)")
        print("Enter '0' to exit shop")

    def buy_item(self, player, item_index):
        global prompt_accumulator
        if 0 <= item_index < len(self.items):
            item = self.items[item_index]
            if not item['sold'] and player.gold >= item['price']:
                player.gold -= item['price']
                item['sold'] = True
                self.apply_item_effect(player, item)
                print(f"Purchase successful! {item['effect']}")
                return True
            elif item['sold']:
                print("This item is sold out!")
            else:
                print("Not enough gold!")
        else:
            print("Invalid item number!")
        return False

class Event:
    def __init__(self, current_floor, player, difficulty):
        self.current_floor = current_floor
        self.player = player
        self.difficulty = difficulty
        self.puzzles = [ 
            {'question': '1+1=?', 'answer': '2'},
            {'question': '1+2=?', 'answer': '3'},
            {'question': '2+2=?', 'answer': '4'},
            {'question': '1+3=?', 'answer': '4'}
        ]
        self.items = [
            {'name': 'Health Potion', 'type': 'health_potion', 'description': 'Restores 50% of max HP'},
            {'name': 'Attack Boost', 'type': 'attack_boost', 'description': 'Permanently increases Attack by 20%'},
            {'name': 'Max HP Boost', 'type': 'health_boost', 'description': 'Permanently increases Max HP by 20%'}
        ]

    def generate_event(self):
        global prompt_accumulator, chat_session
        puzzle = random.choice(self.puzzles)
        
        current_turn_context = prompt_accumulator 
        current_turn_context += f"\n{puzzle['question']}"
        instruction = "Respond with only your numerical answer to the puzzle."

        print(current_turn_context)
        # chat_session.clear_history()
        extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction)
        prompt_accumulator = "" 

        # LLM Raw response is printed in send_message


        answer = ""
        if extracted_action and extracted_action != "Error.":
            answer = extracted_action.strip() # Should be the number

        if answer.lower() == puzzle['answer'].lower():
            prompt_accumulator += "\nPuzzle solved successfully!"
            # Boss hint for all floors
            prompt_accumulator += "\nYou received a hint for the Boss battle:"
            prompt_accumulator += f"\nOn Floor {MAX_FLOOR}, there's a Boss. When its HP drops below 20%, it becomes invincible for 5 turns."
            boss = Boss(MAX_FLOOR, self.difficulty)
            prompt_accumulator += f"\nBoss HP: {boss.hp}"
            prompt_accumulator += f"\nBoss Attack: {boss.attack}"
            prompt_accumulator += f"\nBoss Defense: {boss.defense}"
            prompt_accumulator += f"\nBoss Attack Speed: {boss.attack_speed}"
            
            # Additional rewards for solving puzzle
            reward_type = random.choice(['gold', 'exp', 'item'])
            if reward_type == 'gold':
                gold = random.randint(5, 15)
                prompt_accumulator += f"\nReceived {gold} Gold!"
                self.player.gain_gold(gold)
            elif reward_type == 'exp':
                exp = random.randint(100, 300)
                prompt_accumulator += f"\nReceived {exp} EXP!"
                self.player.gain_exp(exp)
            else:
                item = random.choice(self.items)
                prompt_accumulator += f"\nReceived {item['name']}!"
                if item['type'] == 'attack_boost':
                    old_attack = self.player.attack
                    self.player.attack = round(self.player.attack * 1.2, 1)
                    prompt_accumulator += f"\nAttack increased from {old_attack} to {self.player.attack}!"
                elif item['type'] == 'health_boost':
                    old_max_hp = self.player.max_hp
                    self.player.max_hp = round(self.player.max_hp * 1.2, 1)
                    self.player.hp = self.player.max_hp 
                    prompt_accumulator += f"\nMax HP increased from {old_max_hp} to {self.player.max_hp}!"
                else: 
                    self.player.items.append(item)
        else:
            prompt_accumulator += f"\nPuzzle failed! The correct answer was: {puzzle['answer']}"
            if extracted_action == "Error.":
                 prompt_accumulator += " (Could not get a valid answer from LLM)."
            penalty = random.randint(5, 15)
            self.player.hp = max(1, self.player.hp - penalty)
            prompt_accumulator += f"\nTook {penalty} damage!"

    def get_stat_name(self, stat):
        stat_names = {
            'hp': 'Max HP',
            'attack': 'Attack',
            'attack_speed': 'Attack Speed'
        }
        return stat_names.get(stat, stat)

class Event_Human(Event):
    def __init__(self, current_floor, player, difficulty):
        super().__init__(current_floor, player, difficulty)

    def generate_event(self):
        puzzle = random.choice(self.puzzles)
        
        print(f"\n{puzzle['question']}")
        extracted_action = input("Respond with only your numerical answer to the puzzle: ")
        answer = extracted_action.strip()

        if answer.lower() == puzzle['answer'].lower():
            print("\nPuzzle solved successfully!")
            print("You received a hint for the Boss battle:")
            print(f"On Floor {MAX_FLOOR}, there's a Boss. When its HP drops below 20%, it becomes invincible for 5 turns.")
            boss = Boss_Human(MAX_FLOOR, self.difficulty)
            print(f"Boss HP: {boss.hp}")
            print(f"Boss Attack: {boss.attack}")
            print(f"Boss Defense: {boss.defense}")
            print(f"Boss Attack Speed: {boss.attack_speed}")
            
            # Additional rewards for solving puzzle
            reward_type = random.choice(['gold', 'exp', 'item'])
            if reward_type == 'gold':
                gold = random.randint(5, 15)
                print(f"\nReceived {gold} Gold!")
                self.player.gain_gold(gold)
            elif reward_type == 'exp':
                exp = random.randint(100, 300)
                print(f"\nReceived {exp} EXP!")
                self.player.gain_exp(exp)
            else:
                item = random.choice(self.items)
                print(f"\nReceived {item['name']}!")
                if item['type'] == 'attack_boost':
                    old_attack = self.player.attack
                    self.player.attack = round(self.player.attack * 1.2, 1)
                    print(f"Attack increased from {old_attack} to {self.player.attack}!")
                elif item['type'] == 'health_boost':
                    old_max_hp = self.player.max_hp
                    self.player.max_hp = round(self.player.max_hp * 1.2, 1)
                    self.player.hp = self.player.max_hp 
                    print(f"Max HP increased from {old_max_hp} to {self.player.max_hp}!")
                else: 
                    self.player.items.append(item)
        else:
            print(f"\nPuzzle failed! The correct answer was: {puzzle['answer']}")
            penalty = random.randint(5, 15)
            self.player.hp = max(1, self.player.hp - penalty)
            print(f"Took {penalty} damage!")

class Battle:
    def __init__(self, player, monster, game):
        self.player = player
        self.monster = monster
        self.turn = 0
        self.game = game

    def calculate_attack_probability(self):
        total_speed = self.player.attack_speed + self.monster.attack_speed
        if total_speed == 0: return 0.5, 0.5
        player_prob = self.player.attack_speed / total_speed
        monster_prob = self.monster.attack_speed / total_speed
        return player_prob, monster_prob

    def calculate_damage(self, attacker, defender):
        return max(0, attacker.attack - defender.defense)

    def process_battle(self):
        global prompt_accumulator, chat_session
        prompt_accumulator += "\nBattle starts!"

        while True:
            self.turn += 1
            prompt_accumulator += f"\nTurn {self.turn}"
            self.display_battle_status()
            player_prob, monster_prob = self.calculate_attack_probability()
            attacker = self.player if random.random() < player_prob else self.monster
            defender = self.monster if attacker == self.player else self.player
            damage = self.calculate_damage(attacker, defender)
            skill_message = None
            if attacker == self.player:
                # Check if berserker skill is enabled
                if self.player.berserker_enabled and self.player.skills['berserker']['condition'](self.player):
                    damage = self.player.skills['berserker']['effect'](self.player)
                    skill_message = "Berserker skill triggered! Damage increased!"
                # Check if life steal skill is enabled
                if self.player.life_steal_enabled:
                    heal_amount = round(self.player.skills['life_steal']['effect'](damage), 1)
                    if heal_amount > 0:
                        self.player.hp = min(self.player.max_hp, self.player.hp + heal_amount)
                        life_steal_msg = f"Life Steal triggered! Healed {heal_amount} HP!"
                        if skill_message:
                            skill_message += f" {life_steal_msg}"
                        else:
                            skill_message = life_steal_msg
            actual_damage_dealt = defender.take_damage(damage)
            self.display_attack_result(attacker, defender, actual_damage_dealt, skill_message)

            # Ask AI every 5 turns whether to continue
            if self.turn % 5 == 0 and self.player.hp > 0 and self.monster.hp > 0:
                global show_fight_data
                if show_fight_data:
                    prompt_accumulator += "\n[Battle Data]"
                    prompt_accumulator += f"\nPlayer: HP {self.player.hp:.1f}/{self.player.max_hp}, Attack {self.player.attack:.1f}, Defense {self.player.defense:.1f}, Attack Speed {self.player.attack_speed:.1f}"
                    prompt_accumulator += f"\nMonster: HP {self.monster.hp:.1f}/{self.monster.max_hp}, Attack {self.monster.attack:.1f}, Defense {self.monster.defense:.1f}, Attack Speed {self.monster.attack_speed:.1f}"
                # Calculate current win rate
                try:
                    player_unit = Unit(hp=self.player.hp, attack=self.player.attack, defense=self.player.defense, speed=self.player.attack_speed)
                    monster_unit = Unit(hp=self.monster.hp, attack=self.monster.attack, defense=self.monster.defense, speed=self.monster.attack_speed)
                    win_rate = win_prob_closed_form(player_unit, monster_unit)
                except Exception as e:
                    win_rate = -1  # Calculation failed
                # 1. Print first five rounds of battle information
                print(prompt_accumulator)
                # 2. Print question
                question = "\nDo you want to continue fighting or run away? (c = continue, r = run)"
                print(question)
                # 3. Send to LLM (AI response will be automatically printed)
                instruction = "Your response MUST be a single character: c (continue) or r (run). No other text."
                while True:
                    extracted_action = chat_session.send_message(prompt_accumulator + question, instruction_tip=instruction)
                    if extracted_action and extracted_action.strip().lower() in ['c', 'r']:
                        extracted_action = extracted_action.strip().lower()
                        break
                    prompt_accumulator += "\nInvalid input! Please respond with 'c' to continue or 'r' to run."
                    print("Invalid input! Please respond with 'c' to continue or 'r' to run.")

                # 4. Clear
                prompt_accumulator = ""

                # 5. Record to txt file
                try:
                    global MODEL_NAME
                    log_suffix = '_show' if show_fight_data else ''
                    if MODEL_NAME != 'Null':
                        with open(f"logs/{MODEL_NAME}{log_suffix}.txt", "a", encoding="utf-8") as f:
                            f.write(f"Difficulty {self.game.difficulty.capitalize()}, Game {self.game.game_idx}, Steps {self.game.steps}, Turn {self.turn}, Win Rate: {win_rate:.3f}, Choice: {extracted_action}\n")
                            f.write(f"  Player: HP {self.player.hp:.1f}/{self.player.max_hp}, Attack {self.player.attack:.1f}, Defense {self.player.defense:.1f}, Attack Speed {self.player.attack_speed:.1f}\n")
                            f.write(f"  Monster: HP {self.monster.hp:.1f}/{self.monster.max_hp}, Attack {self.monster.attack:.1f}, Defense {self.monster.defense:.1f}, Attack Speed {self.monster.attack_speed:.1f}\n")
                    elif MODEL_NAME == 'Null':
                        pass
                except Exception as log_e:
                    print(f"Write log.txt failed: {log_e}")
                if extracted_action == 'r':
                    # Monster attacks once
                    run_damage = self.calculate_damage(self.monster, self.player)
                    self.player.take_damage(run_damage)
                    prompt_accumulator += f"\nMonster dealt {run_damage} damage to you as you fled!"
                    if self.player.hp <= 0:
                        prompt_accumulator += "\nYou died while escaping!"
                        raise GameOverException("Player died while escaping!")
                    else:
                        raise GameOverException("Player escaped")
                # Otherwise continue fighting

            if self.player.hp <= 0 or self.monster.hp <= 0:
                if self.player.hp > 0:
                    exp_gained = self.monster.exp_value
                    gold_gained = self.monster.gold_value
                    prompt_accumulator += "\nBattle won!"
                    prompt_accumulator += f"\nGained EXP: {exp_gained}"
                    prompt_accumulator += f"\nGained Gold: {gold_gained}"
                    self.player.gain_exp(exp_gained)
                    self.player.gain_gold(gold_gained)
                    prompt_accumulator += "\nCurrent Stats:"
                    prompt_accumulator += f"\nHP: {round(self.player.hp, 1)}/{self.player.max_hp}"
                    prompt_accumulator += f"\nAttack: {round(self.player.attack, 1)}"
                    prompt_accumulator += f"\nDefense: {round(self.player.defense, 1)}"
                    prompt_accumulator += f"\nAttack Speed: {round(self.player.attack_speed, 1)}"
                    prompt_accumulator += f"\nLevel: {self.player.level}"
                    prompt_accumulator += f"\nEXP: {self.player.exp}/{self.player.exp_to_next_level}"
                    prompt_accumulator += f"\nGold: {self.player.gold}"
                else:
                    prompt_accumulator += "\nBattle lost!"
                    raise GameOverException("Player died, game over!")
                break


    def display_battle_status(self):
        global prompt_accumulator
        prompt_accumulator += f"\nPlayer HP: {round(self.player.hp, 1)}/{self.player.max_hp}"
        prompt_accumulator += f"\n{self.monster.name} HP: {round(self.monster.hp, 1)}/{self.monster.max_hp}"

    def display_attack_result(self, attacker, defender, damage, skill_message=None):
        global prompt_accumulator
        prompt_accumulator += f"\n{attacker.name} dealt {round(damage, 1)} damage to {defender.name}!"
        if skill_message:
            prompt_accumulator += f"\nSkill: {skill_message}"

class Battle_Human(Battle):
    def __init__(self, player, monster, game):
        super().__init__(player, monster, game)
    def process_battle(self):
        print("\nBattle starts!")

        while True:
            self.turn += 1
            print(f"\nTurn {self.turn}")
            self.display_battle_status()
            player_prob, monster_prob = self.calculate_attack_probability()
            attacker = self.player if random.random() < player_prob else self.monster
            defender = self.monster if attacker == self.player else self.player
            damage = self.calculate_damage(attacker, defender)
            skill_message = None
            if attacker == self.player:
                # Check if berserker skill is enabled
                if self.player.berserker_enabled and self.player.skills['berserker']['condition'](self.player):
                    damage = self.player.skills['berserker']['effect'](self.player)
                    skill_message = "Berserker skill triggered! Damage increased!"
                # Check if life steal skill is enabled
                if self.player.life_steal_enabled:
                    heal_amount = round(self.player.skills['life_steal']['effect'](damage), 1)
                    if heal_amount > 0:
                        self.player.hp = min(self.player.max_hp, self.player.hp + heal_amount)
                        life_steal_msg = f"Life Steal triggered! Healed {heal_amount} HP!"
                        if skill_message:
                            skill_message += f" {life_steal_msg}"
                        else:
                            skill_message = life_steal_msg
            actual_damage_dealt = defender.take_damage(damage)
            self.display_attack_result(attacker, defender, actual_damage_dealt, skill_message)

            # Ask AI every 5 rounds if it wants to continue
            if self.turn % 5 == 0 and self.player.hp > 0 and self.monster.hp > 0:
                global show_fight_data
                if show_fight_data:
                    print("\n[Battle Data]")
                    print(f"Player: HP {self.player.hp:.1f}/{self.player.max_hp}, Attack {self.player.attack:.1f}, Defense {self.player.defense:.1f}, Attack Speed {self.player.attack_speed:.1f}")
                    print(f"Monster: HP {self.monster.hp:.1f}/{self.monster.max_hp}, Attack {self.monster.attack:.1f}, Defense {self.monster.defense:.1f}, Attack Speed {self.monster.attack_speed:.1f}")
                # Calculate current win rate
                try:
                    player_unit = Unit(hp=self.player.hp, attack=self.player.attack, defense=self.player.defense, speed=self.player.attack_speed)
                    monster_unit = Unit(hp=self.monster.hp, attack=self.monster.attack, defense=self.monster.defense, speed=self.monster.attack_speed)
                    win_rate = win_prob_closed_form(player_unit, monster_unit)
                except Exception as e:
                    win_rate = -1  # Calculation failed
                question = "\nDo you want to continue fighting or run away? (c = continue, r = run)"
                print(question)
                instruction = "Your response MUST be a single character: c (continue) or r (run). No other text: "
                while True:
                    extracted_action = input(instruction).strip().lower()
                    if extracted_action in ['c', 'r']:
                        break
                    print("Invalid input! Please enter 'c' to continue or 'r' to run.")
                # 5. Record to txt file
                try:
                    log_suffix = '_show' if show_fight_data else ''
                    with open(f"logs/human{log_suffix}.txt", "a", encoding="utf-8") as f:
                        f.write(f"Difficulty {self.game.difficulty.capitalize()}, Game {self.game.game_idx}, Steps {self.game.steps}, Turn {self.turn}, Win Rate: {win_rate:.3f}, Choice: {extracted_action}\n")
                        f.write(f"  Player: HP {self.player.hp:.1f}/{self.player.max_hp}, Attack {self.player.attack:.1f}, Defense {self.player.defense:.1f}, Attack Speed {self.player.attack_speed:.1f}\n")
                        f.write(f"  Monster: HP {self.monster.hp:.1f}/{self.monster.max_hp}, Attack {self.monster.attack:.1f}, Defense {self.monster.defense:.1f}, Attack Speed {self.monster.attack_speed:.1f}\n")
                except Exception as log_e:
                    print(f"Write log.txt failed: {log_e}")
                if extracted_action == 'r':
                    # Monster attacks once
                    run_damage = self.calculate_damage(self.monster, self.player)
                    self.player.take_damage(run_damage)
                    print(f"\nMonster dealt {run_damage} damage to you as you fled!")
                    if self.player.hp <= 0:
                        print("\nYou died while escaping!")
                        raise GameOverException("Player died while escaping!")
                    else:
                        raise GameOverException("Player escaped")
                # Otherwise continue fighting

            if self.player.hp <= 0 or self.monster.hp <= 0:
                if self.player.hp > 0:
                    exp_gained = self.monster.exp_value
                    gold_gained = self.monster.gold_value
                    print("\nBattle won!")
                    print(f"Gained EXP: {exp_gained}")
                    print(f"Gained Gold: {gold_gained}")
                    self.player.gain_exp(exp_gained)
                    self.player.gain_gold(gold_gained)
                    print("\nCurrent Stats:")
                    print(f"HP: {round(self.player.hp, 1)}/{self.player.max_hp}")
                    print(f"Attack: {round(self.player.attack, 1)}")
                    print(f"Defense: {round(self.player.defense, 1)}")
                    print(f"Attack Speed: {round(self.player.attack_speed, 1)}")
                    print(f"Level: {self.player.level}")
                    print(f"EXP: {self.player.exp}/{self.player.exp_to_next_level}")
                    print(f"Gold: {self.player.gold}")
                else:
                    print("\nBattle lost!")
                    raise GameOverException("Player died, game over!")
                break

    def display_battle_status(self):
        print(f"Player HP: {round(self.player.hp, 1)}/{self.player.max_hp}")
        print(f"{self.monster.name} HP: {round(self.monster.hp, 1)}/{self.monster.max_hp}")

    def display_attack_result(self, attacker, defender, damage, skill_message=None):
        print(f"{attacker.name} dealt {round(damage, 1)} damage to {defender.name}!")
        if skill_message:
            print(f"Skill: {skill_message}")

class Game:
    def __init__(self, difficulty, game_idx):
        self.current_floor = 1
        self.difficulty = difficulty
        self.game_idx = game_idx
        self.player = Player(difficulty)
        self.map = Map(self.current_floor)
        self.shop = Shop()
        self.event = Event(self.current_floor, self.player, self.difficulty)
        self.player.position = self.map.entry_pos
        self.steps = 0
        self.map.update_visibility(self.player.position, self.get_vision_range())
        self.last_positions = []  # New: Used to record positions from last three rounds

    def get_vision_range(self):
        if self.difficulty == 'easy':
            return self.map.size 
        elif self.difficulty == 'normal':
            return self.map.size 
        elif self.difficulty == 'hard':
            return 1 
        else: 
            return 1 

    def display_game_status(self):
        global prompt_accumulator
        prompt_accumulator += f"\nSteps already taken: {self.steps}"
        self.map.display_map(self.player.position, self.get_vision_range())
        prompt_accumulator += "M:Monster\n?:Puzzle\nS:Shop\nY:You\nO:empty space\nX:unexplored area\n"
        # New: Current position and last three positions
        prompt_accumulator += f"Current position: {self.player.position}\n"
        if len(self.last_positions) >= 3:
            last3 = self.last_positions[-3:]
        else:
            last3 = self.last_positions
        prompt_accumulator += "Last 3 positions: " + ", ".join(str(pos) for pos in last3) + "\n"

        prompt_accumulator += "\nPlayer Status:"
        prompt_accumulator += f"\nLevel: {self.player.level} (EXP: {self.player.exp}/{self.player.exp_to_next_level})"
        prompt_accumulator += f"\nHP: {round(self.player.hp, 1)}/{self.player.max_hp}"
        prompt_accumulator += f"\nAttack: {round(self.player.attack, 1)}"
        prompt_accumulator += f"\nDefense: {round(self.player.defense, 1)}"
        prompt_accumulator += f"\nAttack Speed: {round(self.player.attack_speed, 1)}"
        prompt_accumulator += f"\nGold: {self.player.gold}"

        prompt_accumulator += "\n\nAvailable Actions:"
        prompt_accumulator += "\n1. Move (w/a/s/d)"
        prompt_accumulator += "\n2. Use Item (i)"
        prompt_accumulator += "\n3. Quit Game (q)"
        prompt_accumulator += "\n4. Read Guide (g)"




        prompt_accumulator += "\nNote 1: Don't walk around aimlessly, otherwise your score will be reduced."
        prompt_accumulator += "\nNote 2: You should kill monsters to improve your stats, otherwise you will be killed by stronger monsters in higher floors."

    def handle_movement(self, direction):
        global prompt_accumulator
        
        old_player_pos = self.player.position
        new_pos_list = list(self.player.position)

        map_height = 1 if self.map.floor % MAX_FLOOR == 0 else self.map.size # Max row index for current map
        map_width = self.map.size # Max col index for current map

        if self.map.floor % MAX_FLOOR == 0: 
            if direction == 'a': new_pos_list[1] -= 1
            elif direction == 'd': new_pos_list[1] += 1
            elif direction in ['w', 's']:
                prompt_accumulator += "\nInvalid move: On Boss floors, you can only move left (a) or right (d)."
                return False 
        else: 
            if direction == 'w': new_pos_list[0] -= 1
            elif direction == 's': new_pos_list[0] += 1
            elif direction == 'a': new_pos_list[1] -= 1
            elif direction == 'd': new_pos_list[1] += 1
        
        new_pos_tuple = tuple(new_pos_list)
        
        # New: Record current position to last_positions
        if not hasattr(self, 'last_positions'):
            self.last_positions = []
        self.last_positions.append(self.player.position)
        if len(self.last_positions) > 10:
            self.last_positions = self.last_positions[-10:]
        if 0 <= new_pos_list[0] < map_height and 0 <= new_pos_list[1] < map_width:
            self.map.update_visibility(new_pos_tuple, self.get_vision_range())
            cell_content = self.map.grid[new_pos_list[0]][new_pos_list[1]]

            self.map.grid[old_player_pos[0]][old_player_pos[1]] = 'O'
            self.player.position = new_pos_tuple
            self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'

            if cell_content == 'M':
                monster = self.generate_monster()
                battle = Battle(self.player, monster, self)
                try:
                    battle.process_battle()
                    if self.player.hp > 0:
                        # Battle victory, clear monster cell
                        self.map.grid[self.player.position[0]][self.player.position[1]] = 'O'
                        self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
                except GameOverException as e:
                    msg = str(e)
                    if "escaped" in msg:
                        # Escape successful, return to original position
                        self.map.grid[self.player.position[0]][self.player.position[1]] = cell_content
                        self.player.position = old_player_pos
                        self.map.grid[old_player_pos[0]][old_player_pos[1]] = 'Y'
                        return
                    elif "died" in msg:
                        return "death"
            elif cell_content == 'S':
                self.handle_shop_interaction() 
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'O'
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
            elif cell_content == '?':
                self.event.generate_event()
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'O'
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
            elif cell_content == 'E': 
                if self.current_floor == MAX_FLOOR: 
                    return "victory"
                else:
                    self.current_floor += 1
                    prompt_accumulator += f"\nEntered Floor {self.current_floor}!"
                    self.map = Map(self.current_floor)
                    self.event = Event(self.current_floor, self.player, self.difficulty)
                    self.shop.refresh_shop()
                    self.player.position = self.map.entry_pos
                    self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
                    self.map.update_visibility(self.player.position, self.get_vision_range())
                    return False
            
            self.steps += 1
            # v3: set a maximum number of actions, beyond which the attempt is considered a failure
            if self.steps > MAX_STEPS:
                prompt_accumulator += f"\nYour steps have exceeded the maximum number ({MAX_STEPS}), game over."
                return "death"
        else:
            prompt_accumulator += "\nCannot move outside the map!"
        return False 

    def handle_shop_interaction(self): 
        global prompt_accumulator, chat_session
        self.shop.display_shop() 

        while True:
            current_turn_context = prompt_accumulator 
            instruction = "Respond with item number (1-5), 'r' to refresh, or '0' to exit."
            
            print(current_turn_context) # Display the shop and prompt to human
            # chat_session.clear_history() 
            extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction)
            prompt_accumulator = "" 

            # LLM Raw Response is printed in send_message


            choice = ""
            if extracted_action and extracted_action != "Error.":
                choice = extracted_action.strip().lower() # Should be '0'-'5', or 'r'

            if choice == '0':
                prompt_accumulator += "Exited shop."
                break
            elif choice == 'r':
                if self.player.gold >= self.shop.refresh_cost:
                    self.player.gold -= self.shop.refresh_cost
                    self.shop.refresh_shop()
                    prompt_accumulator += "Shop refreshed.\n"
                    self.shop.display_shop() 
                else:
                    prompt_accumulator += f"Not enough gold to refresh! Need {self.shop.refresh_cost} Gold.\n"
                    self.shop.display_shop() 
            else:
                try:
                    # Allow LLM to respond with "1." or similar, extract digit
                    item_number_str = ''.join(filter(str.isdigit, choice))
                    if not item_number_str:
                        if choice: # If choice is not empty but not a digit (e.g. LLM said "buy first item")
                             prompt_accumulator += f"Invalid input '{choice}'. Please use a number, 'r', or '0'.\n"
                        else: # Choice was empty
                             prompt_accumulator += "No valid input received.\n"
                        self.shop.display_shop()
                        continue
                    
                    item_index = int(item_number_str) - 1
                    if self.shop.buy_item(self.player, item_index):
                        prompt_accumulator += "\n" 
                        self.shop.display_shop()
                    else:
                        prompt_accumulator += "\n"
                        self.shop.display_shop()
                except ValueError: # Should be caught by isdigit check mostly
                    prompt_accumulator += f"Invalid input '{choice}'.\n"
                    self.shop.display_shop()

    def handle_item_use(self):
        global prompt_accumulator, chat_session
        if not self.player.items:
            prompt_accumulator += "Backpack is empty!"
            return

        # Build context for LLM
        current_turn_context = prompt_accumulator # Start with any previous general context
        current_turn_context += "\nBackpack Items:"
        if not self.player.items: # Double check, though covered above
             current_turn_context += "\n(Empty)"
        else:
            for i, item in enumerate(self.player.items, 1):
                current_turn_context += f"\n{i}. {item['name']}"
        
        current_turn_context += "\nSelect item number to use (0 to return):"
        instruction = "Respond with the item number (e.g., 1) or 0 to return."

        print(current_turn_context) # Show item list and prompt to human
        # chat_session.clear_history()
        extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction)
        prompt_accumulator = ""

        # LLM Raw response is printed in send_message


        if extracted_action == "Error." or not extracted_action:
            prompt_accumulator += "Error receiving item choice from LLM."
            return

        try:
            choice_str = ''.join(filter(str.isdigit, extracted_action.strip()))
            if not choice_str:
                prompt_accumulator += f"Invalid item choice '{extracted_action}' from LLM."
                return

            choice = int(choice_str)
            if choice == 0:
                prompt_accumulator += "Returning to game."
                return
            if 1 <= choice <= len(self.player.items):
                self.player.use_item(choice - 1) 
            else:
                prompt_accumulator += f"Invalid item choice '{choice}'. Number out of range."
        except ValueError:
            prompt_accumulator += f"LLM response '{extracted_action}' was not a valid number for item choice."

    def generate_monster(self):
        if self.current_floor % MAX_FLOOR == 0:
            return Boss(self.current_floor, self.difficulty)
        else:
            return Monster(self.current_floor, self.difficulty)

    def get_current_floor_monster_info(self):
        temp_monster = Monster(self.current_floor, self.difficulty)
        return f"HP: {temp_monster.hp}, Attack: {temp_monster.attack}, Defense: {temp_monster.defense}, Attack Speed: {round(temp_monster.attack_speed,1)}"

    def display_guide(self):
        guide = "\nGame Guide:\n"
        guide += "Brave adventurer, welcome to the Magic Tower.\n"
        guide += "You will escape by exploring the map, improving your stats, and defeating monsters.\n"
        guide += f"The Magic Tower has {MAX_FLOOR} floors. You are currently on Floor {self.current_floor}.\n"
        
        if self.current_floor % MAX_FLOOR != 0:
            guide += "Monsters on this floor have approximately these stats:\n"
            guide += f"{self.get_current_floor_monster_info()}\n"
        else:
            temp_boss = Boss(self.current_floor, self.difficulty)
            guide += f"You have reached the Boss area of Floor {self.current_floor}!\n"
            guide += f"Boss Stats: HP: {temp_boss.hp}, Attack: {temp_boss.attack}, Defense: {temp_boss.defense}, Attack Speed: {round(temp_boss.attack_speed,1)}\n"

        guide += "\nControls\n"
        guide += "Each turn, you can enter a letter for an action:\n"
        guide += "1. Move (w/a/s/d) - one unit per move\n"
        guide += "2. Use Item (i)\n"
        guide += "3. Quit Game (q)\n"
        guide += "4. Read Guide (g)\n"
        guide += "\nMap Information\n"
        guide += "Each floor's top-left corner is the entrance, bottom-right corner is Exit (E). Entering E takes you to the next floor, but you cannot return to previous floors.\n"
        guide += "Other cells are randomly distributed as follows:\n"
        guide += "1. 3 Monsters (M), entering M triggers automatic battle\n"
        guide += "2. 1 Puzzle (?), solve it for rewards, fail for penalties\n"
        guide += "3. 1 Shop (S), enter to buy items\n"
        guide += "4. You (Y), shows your current position\n"
        guide += "5. O, empty space\n"
        guide += "6. X, unexplored area, only exists in Hard and Insane modes\n"
        guide += "\nShop System\n"
        guide += "In the shop, enter numbers 1-5 to buy items, r to refresh shop, 0 to exit\n"
        guide += "Each item costs 10 gold, refreshing costs 10 gold\n"
        guide += "After exiting, the shop disappears, changing from S to O. To enter a shop again, you must go to the next floor\n"
        guide += "\nBattle System\n"
        guide += "Entering M automatically triggers battle with Monster\n"
        guide += "Attack probability ratio equals attack speed ratio\n"
        guide += "For example, if A's attack speed is 2 and B's is 1, each turn has 2/3 chance A attacks B, 1/3 chance B attacks A\n"
        guide += "Damage dealt = Attack - Defense, minimum damage is 0\n"
        guide += "At 0 HP, death occurs. If you die, game over. If monster dies, you gain experience and gold\n"
        guide += "\nLevel System\n"
        guide += "Player gains a level when experience reaches 1000, choose one attribute to improve\n"
        guide += "Options 1-3 are randomly generated, choose based on your situation\n"
        guide += "\nNote: Monsters get stronger with each floor, make sure you're strong enough before advancing, don't rush!"
        guide += "\nNote: Although avoiding Monsters is safe, you also need to fight them to improve your attributes, otherwise you will be killed by the Boss. So don't just avoid them all the time."
        return guide

class Game_Human(Game):
    def __init__(self, difficulty, game_idx=0):
        self.current_floor = 1
        self.difficulty = difficulty
        self.game_idx = game_idx
        self.player = Player_Human(difficulty)
        self.map = Map_Human(self.current_floor)
        self.shop = Shop_Human()
        self.event = Event_Human(self.current_floor, self.player, self.difficulty)
        self.player.position = self.map.entry_pos
        self.steps = 0
        self.map.update_visibility(self.player.position, self.get_vision_range())
        self.last_positions = []  # New: Used to record positions from last three rounds

    def display_game_status(self):
        print(f"\nSteps already taken: {self.steps}")
        self.map.display_map(self.player.position, self.get_vision_range())
        print("M:Monster\n?:Puzzle\nS:Shop\nY:You\nO:empty space\nX:unexplored area")
        # New: Current position and last three positions
        print(f"Current position: {self.player.position}")
        if len(self.last_positions) >= 3:
            last3 = self.last_positions[-3:]
        else:
            last3 = self.last_positions
        print("Last 3 positions: " + ", ".join(str(pos) for pos in last3))

        print("\nPlayer Status:")
        print(f"Level: {self.player.level} (EXP: {self.player.exp}/{self.player.exp_to_next_level})")
        print(f"HP: {round(self.player.hp, 1)}/{self.player.max_hp}")
        print(f"Attack: {round(self.player.attack, 1)}")
        print(f"Defense: {round(self.player.defense, 1)}")
        print(f"Attack Speed: {round(self.player.attack_speed, 1)}")
        print(f"Gold: {self.player.gold}")

        print("\nAvailable Actions:")
        print("1. Move (w/a/s/d)")
        print("2. Use Item (i)")
        print("3. Quit Game (q)")
        print("4. Read Guide (g)")

        print("\nNote 1: Don't walk around aimlessly, otherwise your score will be reduced.")
        print("Note 2: You should kill monsters to improve your stats, otherwise you will be killed by stronger monsters in higher floors.")

    def handle_movement(self, direction):
        old_player_pos = self.player.position
        new_pos_list = list(self.player.position)

        map_height = 1 if self.map.floor % MAX_FLOOR == 0 else self.map.size # Max row index for current map
        map_width = self.map.size # Max col index for current map

        if self.map.floor % MAX_FLOOR == 0: 
            if direction == 'a': new_pos_list[1] -= 1
            elif direction == 'd': new_pos_list[1] += 1
            elif direction in ['w', 's']:
                print("\nInvalid move: On Boss floors, you can only move left (a) or right (d).")
                return False 
        else: 
            if direction == 'w': new_pos_list[0] -= 1
            elif direction == 's': new_pos_list[0] += 1
            elif direction == 'a': new_pos_list[1] -= 1
            elif direction == 'd': new_pos_list[1] += 1
        
        new_pos_tuple = tuple(new_pos_list)
        
        # New: Record current position to last_positions
        if not hasattr(self, 'last_positions'):
            self.last_positions = []
        self.last_positions.append(self.player.position)
        if len(self.last_positions) > 10:
            self.last_positions = self.last_positions[-10:]
        if 0 <= new_pos_list[0] < map_height and 0 <= new_pos_list[1] < map_width:
            self.map.update_visibility(new_pos_tuple, self.get_vision_range())
            cell_content = self.map.grid[new_pos_list[0]][new_pos_list[1]]

            self.map.grid[old_player_pos[0]][old_player_pos[1]] = 'O'
            self.player.position = new_pos_tuple
            self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'

            if cell_content == 'M':
                monster = self.generate_monster()
                battle = Battle_Human(self.player, monster, self)
                try:
                    battle.process_battle()
                    if self.player.hp > 0:
                        # Battle victory, clear monster cell
                        self.map.grid[self.player.position[0]][self.player.position[1]] = 'O'
                        self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
                except GameOverException as e:
                    msg = str(e)
                    if "escaped" in msg:
                        # Escape successful, return to original position
                        self.map.grid[self.player.position[0]][self.player.position[1]] = cell_content
                        self.player.position = old_player_pos
                        self.map.grid[old_player_pos[0]][old_player_pos[1]] = 'Y'
                        return
                    elif "died" in msg:
                        return "death"
            elif cell_content == 'S':
                self.handle_shop_interaction() 
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'O'
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
            elif cell_content == '?':
                self.event.generate_event()
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'O'
                self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
            elif cell_content == 'E': 
                if self.current_floor == MAX_FLOOR: 
                    return "victory"
                else:
                    self.current_floor += 1
                    print(f"\nEntered Floor {self.current_floor}!")
                    self.map = Map_Human(self.current_floor)
                    self.event = Event_Human(self.current_floor, self.player, self.difficulty)
                    self.shop.refresh_shop()
                    self.player.position = self.map.entry_pos
                    self.map.grid[self.player.position[0]][self.player.position[1]] = 'Y'
                    self.map.update_visibility(self.player.position, self.get_vision_range())
                    return False
            
            self.steps += 1
            # v3: set a maximum number of actions, beyond which the attempt is considered a failure
            if self.steps > MAX_STEPS:
                print(f"\nYour steps have exceeded the maximum number ({MAX_STEPS}), game over.")
                return "death"
        else:
            print("\nCannot move outside the map!")
        return False 

    def handle_shop_interaction(self): 
        self.shop.display_shop() 

        while True:
            instruction = "Respond with item number (1-5), 'r' to refresh, or '0' to exit: "
            
            extracted_action = input(instruction)

            choice = ""
            if extracted_action and extracted_action != "Error.":
                choice = extracted_action.strip().lower() # Should be '0'-'5', or 'r'

            if choice == '0':
                print("Exited shop.")
                break
            elif choice == 'r':
                if self.player.gold >= self.shop.refresh_cost:
                    self.player.gold -= self.shop.refresh_cost
                    self.shop.refresh_shop()
                    print("Shop refreshed.\n")
                    self.shop.display_shop() 
                else:
                    print(f"Not enough gold to refresh! Need {self.shop.refresh_cost} Gold.\n")
                    self.shop.display_shop() 
            else:
                try:
                    # Allow LLM to respond with "1." or similar, extract digit
                    item_number_str = ''.join(filter(str.isdigit, choice))
                    if not item_number_str:
                        if choice: # If choice is not empty but not a digit (e.g. LLM said "buy first item")
                            print(f"Invalid input '{choice}'. Please use a number, 'r', or '0'.\n")
                        else: # Choice was empty
                            print("No valid input received.\n")
                        self.shop.display_shop()
                        continue
                    
                    item_index = int(item_number_str) - 1
                    if self.shop.buy_item(self.player, item_index):
                        print("\n")
                        self.shop.display_shop()
                    else:
                        print("\n")
                        self.shop.display_shop()
                except ValueError: # Should be caught by isdigit check mostly
                    print(f"Invalid input '{choice}'.\n")
                    self.shop.display_shop()

    def handle_item_use(self):
        if not self.player.items:
            print("Backpack is empty!")
            return

        print("\nBackpack Items:")
        if not self.player.items: # Double check, though covered above
             current_turn_context += "\n(Empty)"
        else:
            for i, item in enumerate(self.player.items, 1):
                print(f"\n{i}. {item['name']}")
        
        print("\nSelect item number to use (0 to return):")
        
        extracted_action = input("Respond with the item number (e.g., 1) or 0 to return: ")

        try:
            choice_str = ''.join(filter(str.isdigit, extracted_action.strip()))
            if not choice_str:
                print(f"Invalid item choice '{extracted_action}'.")
                return

            choice = int(choice_str)
            if choice == 0:
                print("Returning to game.")
                return
            if 1 <= choice <= len(self.player.items):
                self.player.use_item(choice - 1) 
            else:
                print(f"Invalid item choice '{choice}'. Number out of range.")
        except ValueError:
            print(f"'{extracted_action}' was not a valid number for item choice.")

    def generate_monster(self):
        if self.current_floor % MAX_FLOOR == 0:
            return Boss_Human(self.current_floor, self.difficulty)
        else:
            return Monster(self.current_floor, self.difficulty)

@dataclass(frozen=True)
class Unit:
    hp: float
    attack: float
    defense: float
    speed: float

    def damage_to(self, other: "Unit") -> float:
        return max(0.0, self.attack - other.defense)

def win_prob_closed_form(a: Unit, b: Unit) -> float:
    dmg_a = a.damage_to(b)
    dmg_b = b.damage_to(a)

    if dmg_a <=0 and dmg_b <= 0:
        raise ValueError("No damage can be dealt")

    if dmg_a <= 0:
        return 0.0

    if dmg_b <= 0:
        return 1.0

    # To win, A need to attack B n_a times
    n_a = int(ceil(b.hp / dmg_a))
    n_b = int(ceil(a.hp / dmg_b))

    # Consider a Bernoulli event
    # flip a coin, p is the probability of face up, q is the probability of face down
    p = a.speed / (a.speed + b.speed)
    q = 1.0 - p

    # Our goal is to calculate the probability of A winning
    # That is, the probability of A attacking B n_a times and B attacking A less than n_b times
    # Suppose end in n_a + k rounds.
    # Before end, A attacks B n_a - 1 times, B attacks A k times (k < n_b)
    # Write the combination of the events: C(n_a + k - 1, k)
    # Last round must be A attacking B
    # prob = \sum_{k=0}^{n_b-1} C(n_a + k - 1, k) \cdot p^{n_a} \cdot q^{k}
    # This is actually the summation of PMF of Negative Binomial Distribution (When A wins, B may attack 0, 1, ..., n_b -1 times, and we can add all probabilities as A wins probability)

    prob = 0.0
    term = p ** n_a # k = 0
    prob += term

    for k in range(1, n_b): # k = 1, 2, ..., n_b - 1
        # C (a + 1, b + 1) = C (a , b) * (a + 1) / (b + 1)
        term *= (n_a + k - 1) / k * q
        prob += term

    return prob

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def write_game_result(result_type, model, difficulty, steps, final_floor, game_idx):
    try:
        with open('data.txt', 'a+', encoding='utf-8') as f:
            f.seek(0)
            lines = f.readlines()
            idx = 1
            for line in lines:
                match = re.match(r'^(\d+):', line.strip())
                if match:
                    try:
                        idx = max(idx, int(match.group(1)) + 1)
                    except:
                        pass
            f.write(f"{idx}: {result_type}\n")
            f.write(f"    Model: {model}\n")
            f.write(f"    Show battle data: {show_fight_data}\n")
            f.write(f"    Difficulty: {difficulty}\n")
            f.write(f"    Game index: {game_idx}\n")
            f.write(f"    Steps: {steps} \n")
            f.write(f"    Final floor: {final_floor}\n\n")
    except Exception as e:
        print(f"[Warning] Failed to write game result: {e}")



def main():
    global prompt_accumulator, chat_session, MODEL_NAME, mode, MAX_STEPS, show_fight_data
    MAX_STEPS = 50
    show_fight_data = False

    print("Please choose mode:")
    print("1. Human Player")
    print("2. Ollama")
    print("3. Gemini")
    print("4. Openai")  # New
    while True:
        try:
            mode_str = input("Please choose (1-4): ")
            mode = int(mode_str)
            if 1 <= mode <= 4:
                break
            print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")

    print("Please choose whether to show battle data:")
    print("1. Yes")
    print("2. No")
    while True:
        try:
            show_fight_data_str = input("Please choose (1-2): ")
            show_fight_data = int(show_fight_data_str)
            if 1 <= show_fight_data <= 2:
                break
            print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")
    if show_fight_data == 1:
        show_fight_data = True
    else:
        show_fight_data = False

    # Ollama mode
    if mode == 2:
        print("Choose AI model: ")
        print("1. qwen3:30b")
        print("2. llama3.3:70b-instruct-q4_K_M")
        print("3. deepseek-r1:latest")
        print("4. gemma3:27b")
        
        while True:
            try:
                model_str = input("Please choose (1-4): ")
                model_choice = int(model_str)
                if 1 <= model_choice <= 4:
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        if model_choice == 1:
            MODEL_NAME = 'qwen3:30b'
        elif model_choice == 2:
            MODEL_NAME = 'llama3.3:70b-instruct-q4_K_M'
        elif model_choice == 3:
            MODEL_NAME = 'deepseek-r1:latest'
        elif model_choice == 4:
            MODEL_NAME = 'gemma3:27b'        
  

        print(f"Attempting to use Ollama model: {MODEL_NAME}")
        # You might want to add a check here to see if Ollama server is running
        # or if the model is available, e.g., by making a simple API call.
        # --- Initialize chat_session only once ---
        chat_session = OllamaChat(model_name=MODEL_NAME)
        
        try:
            requests.get(f"{chat_session.base_url}/api/tags", timeout=2).raise_for_status()
            print("Ollama server seems to be running.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Ollama server not reachable at {chat_session.base_url}. Please ensure Ollama is running.")
            print(f"Details: {e}")
            return


        welcome_message = "Welcome to the Magic Tower!\n\nPlease choose difficulty:\n1. Easy\n2. Normal\n3. Hard\n4. Insane"
        print(welcome_message)

        while True:
            try:
                choice_str = input("\nPlease choose (1-4): ")
                choice = int(choice_str)
                if 1 <= choice <= 4:
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")

        difficulties = ['easy', 'normal', 'hard', 'insane']

        for game_idx in range(MAX_GAMES):
            prompt_accumulator = "" # Clear each game
            prompt_accumulator += f"===== Game {game_idx+1}/{MAX_GAMES} =====\n"
            game = Game(difficulties[choice-1], game_idx + 1)
            # Initial guidance
            prompt_accumulator += "\nYou awaken in darkness, finding yourself in a mysterious room with a small booklet. You open it and find a guide:\n"
            prompt_accumulator += game.display_guide()
            prompt_accumulator += "\nYou close the booklet and begin to explore the Magic Tower."
            initial_instruction = "The LLM will act as the player. Respond with a single character like 'k' or 'c' to acknowledge and continue."
            initial_context_for_llm = prompt_accumulator + f"\n\n({initial_instruction})"
            print(initial_context_for_llm)
            extracted_action = chat_session.send_message(initial_context_for_llm)
            prompt_accumulator = ""
            while True:
                game.display_game_status()
                current_turn_context = prompt_accumulator
                current_turn_context += "\nEnter your action (you only need to input a single letter): "
                instruction_for_action = "Your response MUST be a single character: w, a, s, d, i, q, or g. No other text."
                print(current_turn_context)
                extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction_for_action)
                prompt_accumulator = ""
                action_to_perform = ""
                if extracted_action and extracted_action != "Error.":
                    action_to_perform = extracted_action
                else:
                    action_to_perform = input("LLM error or no response. Manual action (w,a,s,d,i,q,g): ").strip().lower()
                    if action_to_perform:
                        action_to_perform = action_to_perform[0]
                    else:
                        action_to_perform = " "
                if action_to_perform == 'q':
                    prompt_accumulator += "\nGame Over!"
                    print(prompt_accumulator)
                    write_game_result('Quit', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                    # 让AI确认继续
                    chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                    break
                elif action_to_perform == 'i':
                    game.handle_item_use()
                elif action_to_perform == 'g':
                    prompt_accumulator += game.display_guide()
                elif action_to_perform in ['w', 'a', 's', 'd']:
                    result = game.handle_movement(action_to_perform)
                    if result == "victory":
                        prompt_accumulator += "\nCongratulations! You cleared the game!"
                        print(prompt_accumulator)
                        write_game_result('Win', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        # Let AI confirm to continue
                        chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                        break
                    elif result == "death":
                        print(prompt_accumulator)
                        write_game_result('Die', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        # Let AI confirm to continue
                        chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                        break
                else:
                    prompt_accumulator += f"Invalid operation: '{action_to_perform}'"
        return

    # Gemini mode
    elif mode == 3:
        print("Choose Gemini model: ")
        print("1. gemini-2.5-flash-lite-preview-06-17")
        print("2. gemini-2.0-flash-lite")
        print("3. gemini-2.5-pro")
        print("4. gemini-2.5-flash")
        
        while True:
            try:
                model_str = input("Please choose (1-4): ")
                model_choice = int(model_str)
                if 1 <= model_choice <= 4:
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        if model_choice == 1:
            MODEL_NAME = 'gemini-2.5-flash-lite-preview-06-17'
        elif model_choice == 2:
            MODEL_NAME = 'gemini-2.0-flash-lite'
        elif model_choice == 3: 
            MODEL_NAME = 'gemini-2.5-pro'
        elif model_choice == 4:
            MODEL_NAME = 'gemini-2.5-flash'

        print(f"Attempting to use Gemini model: {MODEL_NAME}")
        welcome_message = "Welcome to the Magic Tower!\n\nPlease choose difficulty:\n1. Easy\n2. Normal\n3. Hard\n4. Insane"
        print(welcome_message)

        while True:
            try:
                choice_str = input("\nPlease choose (1-4): ")
                choice = int(choice_str)
                if 1 <= choice <= 4:
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")

        difficulties = ['easy', 'normal', 'hard', 'insane']
        # --- Initialize chat_session only once ---
        chat_session = GeminiChat(api_key=GEMINI_API_KEY, model_name=MODEL_NAME)
        for game_idx in range(MAX_GAMES):
            prompt_accumulator = ""
            prompt_accumulator += f"===== Game {game_idx+1}/{MAX_GAMES} =====\n"
            game = Game(difficulties[choice-1], game_idx + 1)
            prompt_accumulator += "\nYou awaken in darkness, finding yourself in a mysterious room with a small booklet. You open it and find a guide:\n"
            prompt_accumulator += game.display_guide()
            prompt_accumulator += "\nYou close the booklet and begin to explore the Magic Tower."
            initial_instruction = "The LLM will act as the player. Respond with a single character like 'k' or 'c' to acknowledge and continue."
            initial_context_for_llm = prompt_accumulator + f"\n\n({initial_instruction})"
            print(initial_context_for_llm)
            extracted_action = chat_session.send_message(initial_context_for_llm)
            prompt_accumulator = ""
            while True:
                game.display_game_status()
                current_turn_context = prompt_accumulator
                current_turn_context += "\nEnter your action (you only need to input a single letter): "
                instruction_for_action = "Your response MUST be a single character: w, a, s, d, i, q, or g. No other text."
                print(current_turn_context)
                extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction_for_action)
                prompt_accumulator = ""
                action_to_perform = ""
                if extracted_action and extracted_action != "Error.":
                    action_to_perform = extracted_action
                else:
                    action_to_perform = input("LLM error or no response. Manual action (w,a,s,d,i,q,g): ").strip().lower()
                    if action_to_perform:
                        action_to_perform = action_to_perform[0]
                    else:
                        action_to_perform = " "
                if action_to_perform == 'q':
                    prompt_accumulator += "\nGame Over!"
                    print(prompt_accumulator)
                    write_game_result('Quit', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                    # 让AI确认继续
                    chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                    break
                elif action_to_perform == 'i':
                    game.handle_item_use()
                elif action_to_perform == 'g':
                    prompt_accumulator += game.display_guide()
                elif action_to_perform in ['w', 'a', 's', 'd']:
                    result = game.handle_movement(action_to_perform)
                    if result == "victory":
                        prompt_accumulator += "\nCongratulations! You cleared the game!"
                        print(prompt_accumulator)
                        write_game_result('Win', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        # 让AI确认继续
                        chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                        break
                    elif result == "death":
                        print(prompt_accumulator)
                        write_game_result('Die', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        # 让AI确认继续
                        chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                        break
                else:
                    prompt_accumulator += f"Invalid operation: '{action_to_perform}'"
        return

    # Human mode
    elif mode == 1:
        welcome_message = "Welcome to Magic Tower!\n\nChoose difficulty:\n1. Easy\n2. Normal\n3. Hard\n4. Insane"
        print(welcome_message)
        
        while True:
            try:
                choice = int(input("\nChoose (1-4): "))
                if 1 <= choice <= 4:
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        
        difficulties = ['easy', 'normal', 'hard', 'insane']



        for game_idx in range(MAX_GAMES):
        
            print(f"===== Game {game_idx+1}/{MAX_GAMES} =====")
            game = Game_Human(difficulties[choice-1], game_idx + 1)
            
            print("You awaken in darkness, finding yourself in a mysterious room with a small booklet. You open it and find a guide:")
            print(game.display_guide())
            print("\nYou close the booklet and begin to explore the Magic Tower.")
            input("Press any key to continue...")
            
            while True:
                #clear_screen()
                game.display_game_status()
                
                action = input("\nEnter action: ").lower()
                if action == 'q':
                    print("\nGame Over!")
                    write_game_result('Quit', "Human", difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                    break
                elif action == 'i':
                    game.handle_item_use()
                elif action == 'g':
                    print(game.display_guide())
                elif action in ['w', 'a', 's', 'd']:
                    result = game.handle_movement(action)
                    if result == "victory":
                        print("\nCongratulations! You've completed the game!")
                        write_game_result('Win', "Human", difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        break
                    elif result == "death":
                        write_game_result('Die', "Human", difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        break
                else:
                    print("Invalid action!")

    # Openai mode
    if mode == 4:
        print("Choose OpenAI model: ")
        print(f"1. gpt-4.1-nano-2025-04-14")
        print(f"2. o3-mini-2025-01-31")
        print(f"3. gpt-4o-mini-2024-07-18")

        while True:
            try:
                model_str = input("Please choose (1-3): ")
                model_choice = int(model_str)
                if model_choice == 1:
                    MODEL_NAME = "gpt-4.1-nano-2025-04-14"
                    break
                elif model_choice == 2:
                    MODEL_NAME = "o3-mini-2025-01-31"
                    break
                elif model_choice == 3:
                    MODEL_NAME = "gpt-4o-mini-2024-07-18"
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        if not OPENAI_API_KEY:
            print("Please set the OPENAI_API_KEY environment variable.")
            return
        print(f"Attempting to use OpenAI model: {MODEL_NAME}")
        welcome_message = "Welcome to the Magic Tower!\n\nPlease choose difficulty:\n1. Easy\n2. Normal\n3. Hard\n4. Insane"
        print(welcome_message)
        while True:
            try:
                choice_str = input("\nPlease choose (1-4): ")
                choice = int(choice_str)
                if 1 <= choice <= 4:
                    break
                print("Invalid choice!")
            except ValueError:
                print("Please enter a valid number!")
        difficulties = ['easy', 'normal', 'hard', 'insane']
        # --- Initialize chat_session only once ---
        chat_session = OpenaiChat(api_key=OPENAI_API_KEY, model_name=MODEL_NAME)
        for game_idx in range(MAX_GAMES):
            prompt_accumulator = ""
            prompt_accumulator += f"===== Game {game_idx+1}/{MAX_GAMES} =====\n"
            game = Game(difficulties[choice-1], game_idx + 1)
            prompt_accumulator += "\nYou awaken in darkness, finding yourself in a mysterious room with a small booklet. You open it and find a guide:\n"
            prompt_accumulator += game.display_guide()
            prompt_accumulator += "\nYou close the booklet and begin to explore the Magic Tower."
            initial_instruction = "The LLM will act as the player. Respond with a single character like 'k' or 'c' to acknowledge and continue."
            initial_context_for_llm = prompt_accumulator + f"\n\n({initial_instruction})"
            print(initial_context_for_llm)
            extracted_action = chat_session.send_message(initial_context_for_llm)
            prompt_accumulator = ""
            while True:
                game.display_game_status()
                current_turn_context = prompt_accumulator
                current_turn_context += "\nEnter your action (you only need to input a single letter): "
                instruction_for_action = "Your response MUST be a single character: w, a, s, d, i, q, or g. No other text."
                print(current_turn_context)
                extracted_action = chat_session.send_message(current_turn_context, instruction_tip=instruction_for_action)
                prompt_accumulator = ""
                action_to_perform = ""
                if extracted_action and extracted_action != "Error.":
                    action_to_perform = extracted_action
                else:
                    action_to_perform = input("LLM error or no response. Manual action (w,a,s,d,i,q,g): ").strip().lower()
                    if action_to_perform:
                        action_to_perform = action_to_perform[0]
                    else:
                        action_to_perform = " "
                if action_to_perform == 'q':
                    prompt_accumulator += "\nGame Over!"
                    print(prompt_accumulator)
                    write_game_result('Quit', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                    # Let AI confirm to continue
                    chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                    break
                elif action_to_perform == 'i':
                    game.handle_item_use()
                elif action_to_perform == 'g':
                    prompt_accumulator += game.display_guide()
                elif action_to_perform in ['w', 'a', 's', 'd']:
                    result = game.handle_movement(action_to_perform)
                    if result == "victory":
                        prompt_accumulator += "\nCongratulations! You cleared the game!"
                        print(prompt_accumulator)
                        write_game_result('Win', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        # Let AI confirm to continue
                        chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                        break
                    elif result == "death":
                        print(prompt_accumulator)
                        write_game_result('Die', MODEL_NAME, difficulties[choice-1].capitalize(), game.steps, game.current_floor, game.game_idx)
                        # Let AI confirm to continue
                        chat_session.send_message(prompt_accumulator, instruction_tip="Please reply any letter to continue to the next game.")
                        break
                else:
                    prompt_accumulator += f"Invalid operation: '{action_to_perform}'"
        return

if __name__ == "__main__":
    try:
        main()
    except GameOverException as e:
        print(prompt_accumulator) 
        print(str(e)) 
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nThanks for playing!")







