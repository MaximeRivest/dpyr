"""Documentation stays executable: every ```python block in every guide,
concatenated in order, must run cleanly. Docs drift now fails CI."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

GUIDE_DIR = Path(__file__).parent.parent / "docs" / "guide"


def guides() -> list[Path]:
    return sorted(GUIDE_DIR.glob("*.md"))


@pytest.mark.parametrize("page", guides(), ids=lambda p: p.name)
def test_guide_code_blocks_run(page: Path) -> None:
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    if not blocks:
        pytest.skip("no python blocks")
    script = "\n\n".join(blocks)
    result = subprocess.run(
        [sys.executable, "-"], input=script, capture_output=True, text=True,
        cwd=GUIDE_DIR.parent.parent, timeout=180)
    assert result.returncode == 0, (
        f"{page.name} code blocks failed:\n{result.stderr[-2000:]}")
