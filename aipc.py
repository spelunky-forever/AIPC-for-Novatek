#!/usr/bin/env python3
"""Run the bundled Pi skill, with a fallback for legacy Cline projects."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for relative in (".pi/skills/novatek-toolkit", ".cline/skills/novatek-toolkit"):
        skill = root / relative
        if (skill / "scripts/nvt.py").is_file() and (skill / "SKILL.md").is_file():
            break
    else:
        raise SystemExit("NOVATEK_SKILL_MISSING: launcher=" + str(root / "aipc.py") +
                         "; expected a complete .pi/skills/novatek-toolkit beside this launcher. "
                         "Do not copy assets/templates manually. For LLMs, use the ORIGINAL toolkit launcher "
                         "with llm bootstrap NEW_EMPTY_PROJECT --model-id ORG/MODEL --profile hopeful_carver. "
                         "Preserve this incomplete directory; CNN refresh-skill is not LLM recovery.")
    scripts = skill / "scripts"
    sys.path.insert(0, str(scripts))
    runpy.run_path(str(scripts / "nvt.py"), run_name="__main__")
