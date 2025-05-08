import json
import re
from typing import Optional
from adventure import AdventureGame
from agent import Agent


def run_adventure_game() -> None:
    # Initialize the adventure game and agent
    game = AdventureGame()
    agent = Agent()
    game_output = game.read_output()

    initial_commands = [
        """
OBSERVATION: The game has just started.
KNOWLEDGE: I know the game is called Adventure and has instructions.
GOALS: I'm not sure of my goal yet.
STRATEGY: I should read the instructions.
COMMAND: command({"line": "yes"})
""".strip()
    ]
    while True:
        print(f"---game--\n{game_output.strip()}")

        # Get LLM response
        if initial_commands:
            model_response = initial_commands.pop(0)
        else:
            model_response = agent.prompt(game_output)
        print(f"---player---\n{model_response}")

        command = parse_command(model_response)
        if command is not None:
            # Send to game
            command_text = command["line"]
            game.send_command(command_text)

            # Read the game's response
            game_output = game.read_output().strip().removeprefix(command_text).strip()

            if not game.is_running():
                print("Game has terminated")
                break


def parse_command(input_str: str) -> Optional[dict[str, str]]:
    # Match anything between the parentheses
    match = re.search(r"command\((.*)\)", input_str)
    if match and match.group(1):
        return json.loads(match.group(1))
    return None


try:
    run_adventure_game()
except KeyboardInterrupt:
    print("Program terminated by user")
