"""Test generators

These give simple system responses, intended for testing.
"""

from typing import List

from garak.generators.base import Generator


class Blank(Generator):
    """This generator always returns the empty string."""

    supports_multiple_generations = True
    generator_family_name = "Test"
    name = "Blank"

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> List[str]:
        return [""] * generations_this_call


class Repeat(Generator):
    """This generator returns the input that was posed to it."""

    supports_multiple_generations = True
    generator_family_name = "Test"
    name = "Repeat"

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> List[str]:
        return [prompt] * generations_this_call


class Single(Generator):
    """This generator returns the a fixed string and does not support multiple generations."""

    supports_multiple_generations = False
    generator_family_name = "Test"
    name = "Single"
    test_generation_string = "ELIM"

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> List[str]:
        if generations_this_call == 1:
            return [self.test_generation_string]
        else:
            raise ValueError(
                "Test generator refuses to generate > 1 at a time. Check generation logic"
            )


DEFAULT_CLASS = "Blank"
