import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from typing import List, Dict, TypedDict, Literal


class ToolParameter(TypedDict):
    description: str
    type: str


class Tool(TypedDict):
    name: str
    description: str
    parameters: Dict[str, ToolParameter]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Agent:
    """Class for interacting with an LLM."""

    def __init__(self, model_path: str = "microsoft/Phi-4-mini-instruct") -> None:
        """Initialize the LLM with necessary configuration."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tools: List[Tool] = [
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

        self.messages: List[Message] = [
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
        <|tool|>{json.dumps(self.tools)}<|/tool|>""",
            },
        ]

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.generation_args = {
            "max_new_tokens": 2000,
            "return_full_text": False,
        }

    def prompt(self, game_output: str) -> str:
        """
        Run the LLM with the given game output and return the model's response.

        Args:
            game_output: The text output from the game to be processed by the LLM

        Returns:
            The raw text response from the LLM
        """
        # Add game output to messages
        self.messages.append({"role": "user", "content": game_output})

        # Get LLM response
        output = self.pipe(self.messages, **self.generation_args)
        model_response: str = output[0]["generated_text"].strip()

        # Add response to message history
        self.messages.append({"role": "assistant", "content": model_response})

        return model_response
