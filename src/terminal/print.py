"""This module contains the print terminal functions"""

import sys


def print_error(message) -> None:
    """Prints a message in red to stderr"""
    print(f"\033[91m{message}\033[0m", file=sys.stderr)


def print_warning(message) -> None:
    """Prints a warning message in orange to stderr"""
    print(f"\033[38;5;208m{message}\033[0m", file=sys.stderr)


def print_info(message) -> None:
    """Prints an info message to stdout"""
    print(message)


def print_success(message) -> None:
    """Prints a success message to stdout"""
    print(f"\033[92m{message}\033[0m")
