"""Tests for code extraction and parsing utilities."""

import json
import re

# Code block extraction regex (from universal_solver_search.py)
CODE_BLOCK_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)


def extract_code(generated_text: str) -> str:
    """Extract code from <code> block."""
    m = CODE_BLOCK_RE.search(generated_text or "")
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


class TestCodeExtraction:
    """Test code extraction from LLM completions."""

    def test_extract_code_basic(self):
        """Test basic code extraction."""
        text = """<think>
Some reasoning here.
</think>
<code>
def solve():
    return [0, 1, 2]
</code>
<answer>
Selected: 0, 1, 2
</answer>"""

        code = extract_code(text)
        assert code is not None
        assert "def solve()" in code
        assert "return [0, 1, 2]" in code

    def test_extract_code_multiple_blocks(self):
        """Test extraction when multiple code blocks exist."""
        text = """<code>
# First block
x = 1
</code>
Some text
<code>
# Second block
y = 2
</code>"""

        code = extract_code(text)
        # Regex extracts first match
        assert code is not None
        assert "x = 1" in code

    def test_extract_code_no_block(self):
        """Test extraction when no code block exists."""
        text = "Just some text without code blocks"
        code = extract_code(text)
        assert code is None

    def test_extract_code_empty_block(self):
        """Test extraction with empty code block."""
        text = "<code></code>"
        code = extract_code(text)
        assert code is None or code == ""

    def test_extract_code_case_insensitive(self):
        """Test that code extraction is case-insensitive."""
        text = """<CODE>
def solve():
    pass
</CODE>"""

        code = extract_code(text)
        assert code is not None
        assert "def solve()" in code

    def test_extract_code_with_whitespace(self):
        """Test extraction preserves code structure."""
        text = """<code>
def solve():
    if True:
        return [1, 2, 3]
</code>"""

        code = extract_code(text)
        assert code is not None
        assert "    if True:" in code  # Indentation preserved
        assert "        return" in code


class TestAnswerParsing:
    """Test answer block parsing."""

    def test_parse_answer_selected_format(self):
        """Test parsing 'Selected: 0, 1, 2' format."""
        text = "<answer>Selected: 0, 1, 2</answer>"
        match = re.search(r"Selected:\s*([0-9,\s]+)", text)
        if match:
            selected_str = match.group(1)
            selected = [int(x.strip()) for x in selected_str.split(",")]
            assert selected == [0, 1, 2]

    def test_parse_answer_json_format(self):
        """Test parsing JSON format in answer."""
        text = '<answer>{"selection": {"variables": [0, 1, 2]}}</answer>'

        json_match = re.search(r'\{.*"selection".*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            assert data["selection"]["variables"] == [0, 1, 2]

    def test_parse_answer_missing(self):
        """Test parsing when answer block is missing."""
        text = "No answer block here"
        match = re.search(r"Selected:\s*([0-9,\s]+)", text)
        assert match is None
