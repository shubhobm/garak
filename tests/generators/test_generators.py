# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import pytest

from garak import _plugins
from garak.generators.test import Blank, Repeat, Single

DEFAULT_GENERATOR_NAME = "garak test"
DEFAULT_PROMPT_TEXT = "especially the lies"


GENERATORS = [
    classname for (classname, active) in _plugins.enumerate_plugins("generators")
]


def test_generators_test_blank():
    g = Blank(DEFAULT_GENERATOR_NAME)
    output = g.generate(prompt="test", generations_this_call=5)
    assert output == [
        "",
        "",
        "",
        "",
        "",
    ], "generators.test.Blank with generations=5 should return five empty strings"


def test_generators_test_repeat():
    g = Repeat(DEFAULT_GENERATOR_NAME)
    output = g.generate(prompt=DEFAULT_PROMPT_TEXT, generations_this_call=1)
    assert output == [
        DEFAULT_PROMPT_TEXT
    ], "generators.test.Repeat should send back a list of the posed prompt string"


def test_generators_test_single_one():
    g = Single(DEFAULT_GENERATOR_NAME)
    output = g.generate(prompt="test", generations_this_call=1)
    assert isinstance(
        output, list
    ), "Single generator .generate() should send back a list"
    assert (
        len(output) == 1
    ), "Single.generate() with generations = 1 should send a list of one string"
    assert isinstance(
        output[0], str
    ), "Single generator output list should contain strings"

    output = g._call_model(prompt="test", generations_this_call=1)
    assert isinstance(output, list), "Single generator _call_model should return a list"
    assert (
        len(output) == 1
    ), "_call_model w/ generations_this_call 1 should return a list of length 1"
    assert isinstance(
        output[0], str
    ), "Single generator output list should contain strings"


def test_generators_test_single_many():
    g = Single(DEFAULT_GENERATOR_NAME)
    output = g.generate(prompt="test", generations_this_call=2)
    assert isinstance(
        output, list
    ), "Single generator .generate() should send back a list"
    assert (
        len(output) == 2
    ), "Single.generate() with generations = 2 should send a list of length 2"
    assert isinstance(
        output[0], str
    ), "Single generator output list should contain strings (first position)"
    assert isinstance(
        output[1], str
    ), "Single generator output list should contain strings (second position)"


def test_generators_test_single_too_many():
    g = Single(DEFAULT_GENERATOR_NAME)
    with pytest.raises(ValueError):
        output = g._call_model(prompt="test", generations_this_call=2)
    assert "Single._call_model should refuse to process generations_this_call > 1"


def test_generators_test_blank_one():
    g = Blank(DEFAULT_GENERATOR_NAME)
    output = g.generate(prompt="test", generations_this_call=1)
    assert isinstance(
        output, list
    ), "Blank generator .generate() should send back a list"
    assert (
        len(output) == 1
    ), "Blank generator .generate() w/ generations=1 should return a list of length 1"
    assert isinstance(
        output[0], str
    ), "Blank generator output list should contain strings"
    assert (
        output[0] == ""
    ), "Blank generator .generate() output list should contain strings"


def test_generators_test_blank_many():
    g = Blank(DEFAULT_GENERATOR_NAME)
    output = g.generate(prompt="test", generations_this_call=2)
    assert isinstance(
        output, list
    ), "Blank generator .generate() should send back a list"
    assert (
        len(output) == 2
    ), "Blank generator .generate() w/ generations=2 should return a list of length 2"
    assert isinstance(
        output[0], str
    ), "Blank generator output list should contain strings (first position)"
    assert isinstance(
        output[1], str
    ), "Blank generator output list should contain strings (second position)"
    assert (
        output[0] == ""
    ), "Blank generator .generate() output list should contain strings (first position)"
    assert (
        output[1] == ""
    ), "Blank generator .generate() output list should contain strings (second position)"


@pytest.mark.parametrize("classname", GENERATORS)
def test_generator_structure(classname):

    m = importlib.import_module("garak." + ".".join(classname.split(".")[:-1]))
    g = getattr(m, classname.split(".")[-1])

    # has method _call_model
    assert "_call_model" in dir(
        g
    ), f"generator {classname} must have a method _call_model"
    # _call_model has a generations_this_call param
    assert (
        "generations_this_call" in inspect.signature(g._call_model).parameters
    ), f"{classname}._call_model() must accept parameter generations_this_call"
    assert (
        "prompt" in inspect.signature(g._call_model).parameters
    ), f"{classname}._call_model() must accept parameter prompt"
    # has method generate
    assert "generate" in dir(g), f"generator {classname} must have a method generate"
    # generate has a generations_this_call param
    assert (
        "generations_this_call" in inspect.signature(g.generate).parameters
    ), f"{classname}.generate() must accept parameter generations_this_call"
    assert (
        "prompt" in inspect.signature(g.generate).parameters
    ), f"{classname}.generate() must accept parameter prompt"
    # generate("") w/ empty string doesn't fail, does return list
