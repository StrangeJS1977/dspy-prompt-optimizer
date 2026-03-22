"""
patcher.py — Skriver optimerede prompts tilbage til kildekoden.

Understøtter:
  - Python-filer med VARIABLE = \"\"\"...\"\"\"-strings
  - YAML config-filer
  - JSON config-filer
  - Generiske tekst-filer med markerings-kommentarer
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional


# ── Python-patcher ────────────────────────────────────────────────────────────

class PythonPromptPatcher:
    """
    Finder og erstatter en triple-quoted string variabel i en Python-fil.

    Eksempel::

        patcher = PythonPromptPatcher(
            source_file=Path("chunk_pdf.py"),
            variable_name="SYSTEM_PROMPT",
        )
        patcher.patch(new_prompt)
    """

    def __init__(self, source_file: Path, variable_name: str):
        self.source_file = source_file
        self.variable_name = variable_name

    def read_current(self) -> Optional[str]:
        """Læs nuværende prompt-værdi."""
        content = self.source_file.read_text(encoding="utf-8")
        pattern = rf'{re.escape(self.variable_name)}\s*=\s*"""(.*?)"""'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1) if match else None

    def patch(self, new_prompt: str, dry_run: bool = False) -> bool:
        """
        Erstat prompt i kildefilen.
        Opretter automatisk .bak backup.
        Returnerer True ved succes.
        """
        content = self.source_file.read_text(encoding="utf-8")
        pattern = rf'({re.escape(self.variable_name)}\s*=\s*""")(.*?)(""")'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            raise ValueError(
                f"Variabel '{self.variable_name}' ikke fundet i {self.source_file}"
            )

        if dry_run:
            old_len = len(match.group(2))
            print(f"[dry-run] Ville erstatte {old_len} → {len(new_prompt)} tegn "
                  f"i {self.source_file.name}")
            return True

        # Backup
        backup = self.source_file.with_suffix(self.source_file.suffix + ".bak")
        shutil.copy2(self.source_file, backup)

        new_content = re.sub(
            pattern,
            lambda m: f'{m.group(1)}{new_prompt}{m.group(3)}',
            content,
            count=1,
            flags=re.DOTALL,
        )
        self.source_file.write_text(new_content, encoding="utf-8")
        print(f"  Opdateret: {self.source_file.name}")
        print(f"  Backup:    {backup.name}")
        return True

    def restore(self) -> bool:
        """Gendan fra .bak backup."""
        backup = self.source_file.with_suffix(self.source_file.suffix + ".bak")
        if not backup.exists():
            raise FileNotFoundError(f"Ingen backup fundet: {backup}")
        shutil.copy2(backup, self.source_file)
        print(f"Gendannet fra backup: {backup.name}")
        return True


# ── YAML-patcher ──────────────────────────────────────────────────────────────

class YamlPromptPatcher:
    """
    Finder og erstatter en prompt-nøgle i en YAML-fil.

    Eksempel::

        patcher = YamlPromptPatcher(
            source_file=Path("config.yaml"),
            key_path="llm.system_prompt",
        )
        patcher.patch(new_prompt)
    """

    def __init__(self, source_file: Path, key_path: str):
        self.source_file = source_file
        self.key_path = key_path  # dot-notation: "llm.system_prompt"

    def patch(self, new_prompt: str, dry_run: bool = False) -> bool:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML kræves: pip install pyyaml")

        content = yaml.safe_load(self.source_file.read_text(encoding="utf-8"))
        keys = self.key_path.split(".")

        # Navigate og sæt
        obj = content
        for key in keys[:-1]:
            obj = obj[key]

        if dry_run:
            old = str(obj.get(keys[-1], ""))
            print(f"[dry-run] Ville erstatte {len(old)} → {len(new_prompt)} tegn "
                  f"ved nøgle '{self.key_path}'")
            return True

        backup = self.source_file.with_suffix(".yaml.bak")
        shutil.copy2(self.source_file, backup)

        obj[keys[-1]] = new_prompt
        self.source_file.write_text(
            yaml.dump(content, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        return True


# ── JSON-patcher ──────────────────────────────────────────────────────────────

class JsonPromptPatcher:
    """
    Finder og erstatter en prompt-nøgle i en JSON-fil.

    Eksempel::

        patcher = JsonPromptPatcher(
            source_file=Path("prompts.json"),
            key_path="system.instruction",
        )
        patcher.patch(new_prompt)
    """

    def __init__(self, source_file: Path, key_path: str):
        self.source_file = source_file
        self.key_path = key_path

    def patch(self, new_prompt: str, dry_run: bool = False) -> bool:
        content = json.loads(self.source_file.read_text(encoding="utf-8"))
        keys = self.key_path.split(".")

        obj = content
        for key in keys[:-1]:
            obj = obj[key]

        if dry_run:
            old = str(obj.get(keys[-1], ""))
            print(f"[dry-run] Ville erstatte {len(old)} → {len(new_prompt)} tegn")
            return True

        backup = self.source_file.with_suffix(".json.bak")
        shutil.copy2(self.source_file, backup)

        obj[keys[-1]] = new_prompt
        self.source_file.write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
