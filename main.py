import json
import os
import pty
import re
import subprocess
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

torch.random.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    # quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tools = [
    {
        "name": "command",
        "description": "Type a command into the game.",
        "parameters": {
            "line": {
                "description": "A line of text. MUST be one or two two words, lowercased.",
                "type": "str",
            }
        },
    }
]

messages = [
    {
        "role": "system",
        "content": f"""
        You are the player in a game of Colossal Cave Adventure. You will receive game output from your user.

IMPORTANT: When responding, follow this structure:
1. OBSERVATION: Briefly summarize what you observe in the current game state
2. KNOWLEDGE: List what you know about the game world so far
3. GOALS: Clearly state your current goals (explore, collect items, solve puzzles)
4. STRATEGY: Explain your reasoning for your next action
5. COMMAND: Use the command tool to send your instruction to the game

Always think carefully about your goals and strategy before sending commands. If previous commands didn't work, try different approaches rather than repeating the same actions.

To send commands to the game, you MUST use the command tool interface like this:
<|tool|>{json.dumps(tools)}<|/tool|>""",
    },
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 2000,
    "return_full_text": False,
}


def run_adventure_game() -> None:
    # Start the adventure game using pty
    master, slave = pty.openpty()
    adventure_process = subprocess.Popen(
        ["adventure"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        text=True,
    )
    os.close(slave)
    game_output = read_output(master)

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

        # Add game output to messages
        messages.append({"role": "user", "content": game_output})

        # Get LLM response
        if initial_commands:
            model_response = initial_commands.pop(0)
        else:
            output = pipe(messages, **generation_args)
            model_response = output[0]["generated_text"].strip()
        messages.append({"role": "assistant", "content": model_response})
        print(f"---player---\n{model_response}")

        command = parse_command(model_response)
        if command is not None:
            # Send to game
            os.write(master, f"{command['line']}\n".encode("utf-8"))

            # Read the game's response
            game_output = (
                read_output(master).strip().removeprefix(command["line"]).strip()
            )

            if adventure_process.poll() is not None:
                print("Game has terminated")
                break


def read_output(fd):
    time.sleep(0.1)
    return os.read(fd, 4096).decode("utf-8")


def parse_command(input_str: str) -> dict[str, str]:
    # Match anything between the parentheses
    match = re.search(r"command\((.*)\)", input_str)
    if match and match.group(1):
        return json.loads(match.group(1))
    return None


try:
    run_adventure_game()
except KeyboardInterrupt:
    print("Program terminated by user")
