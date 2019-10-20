#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sknni` package."""


import unittest
from click.testing import CliRunner

from sknni import cli


class TestSknni(unittest.TestCase):
    """Tests for `sknni` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.cli)
        assert result.exit_code == 0
