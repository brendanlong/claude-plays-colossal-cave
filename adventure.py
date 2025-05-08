import os
import pty
import subprocess
import time


class AdventureGame:
    """Interface for interacting with the Colossal Cave Adventure game."""

    def __init__(self) -> None:
        """
        Initialize and connect to the adventure game process.

        Creates a pseudoterminal and starts the adventure game process.
        """
        # Start the adventure game using pty
        self.master, slave = pty.openpty()
        self.adventure_process = subprocess.Popen(
            ["adventure"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            text=True,
        )
        os.close(slave)

    def send_command(self, command: str) -> None:
        """
        Send a command to the adventure game.

        Args:
            command: The command string to send to the game
        """
        os.write(self.master, f"{command}\n".encode("utf-8"))

    def read_output(self) -> str:
        """
        Read output from the adventure game.

        Returns:
            The output text from the game
        """
        time.sleep(0.1)
        return os.read(self.master, 4096).decode("utf-8")

    def is_running(self) -> bool:
        """
        Check if the adventure game process is still running.

        Returns:
            True if the game is running, False otherwise
        """
        return self.adventure_process.poll() is None
