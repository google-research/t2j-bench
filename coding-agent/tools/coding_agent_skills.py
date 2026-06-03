# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

from .workspace import get_workspace_root, to_workspace_relative

_DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[1] / ".gemini" / "skills"


def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
    values: dict[str, str] = {}
    current_key: str | None = None

    for raw_line in frontmatter.splitlines():
        if not raw_line.strip():
            continue

        if not raw_line.startswith((" ", "\t")) and ":" in raw_line:
            key, value = raw_line.split(":", maxsplit=1)
            current_key = key.strip()
            values[current_key] = value.strip().strip('"').strip("'")
            continue

        if current_key is None:
            continue

        continuation = raw_line.strip().strip('"').strip("'")
        if continuation:
            existing = values.get(current_key, "")
            values[current_key] = f"{existing} {continuation}".strip()

    return values


def _read_skill_file(skill_file) -> dict:
    content = skill_file.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise ValueError(f"Skill file is missing YAML frontmatter: {skill_file}")

    parts = content.split("---\n", maxsplit=2)
    if len(parts) < 3:
        raise ValueError(f"Skill file has invalid frontmatter: {skill_file}")

    frontmatter = parts[1]
    body = parts[2].lstrip()
    metadata = _parse_frontmatter(frontmatter)

    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()
    if not name or not description:
        raise ValueError(f"Skill file must define name and description: {skill_file}")

    resource_files = sorted(
        to_workspace_relative(path)
        for path in skill_file.parent.rglob("*")
        if path.is_file() and path.name != "SKILL.md"
    )

    return {
        "name": name,
        "description": description,
        "path": to_workspace_relative(skill_file),
        "directory": to_workspace_relative(skill_file.parent),
        "body": body,
        "resource_files": resource_files,
    }


def _candidate_skills_roots() -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    candidates = [
        (get_workspace_root() / ".gemini" / "skills", "workspace"),
        (_DEFAULT_SKILLS_ROOT, "agent"),
    ]
    for path, source in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        roots.append((resolved, source))
    return roots


def _discover_skills() -> list[dict]:
    discovered_by_name: dict[str, dict] = {}

    for skills_root, source in _candidate_skills_roots():
        for skill_file in sorted(skills_root.glob("*/SKILL.md")):
            try:
                skill = _read_skill_file(skill_file)
                skill["source"] = source
                skill["skills_root"] = str(skills_root)
            except Exception as exc:
                skill = {
                    "name": skill_file.parent.name,
                    "description": f"Invalid skill definition: {exc}",
                    "path": to_workspace_relative(skill_file),
                    "directory": to_workspace_relative(skill_file.parent),
                    "invalid": True,
                    "source": source,
                    "skills_root": str(skills_root),
                }

            discovered_by_name.setdefault(skill["name"], skill)

    return list(discovered_by_name.values())


def list_skills() -> dict:
    """List discovered workspace skills without loading their full contents.

    Return lightweight metadata first so the agent can decide whether a skill is
    relevant before loading its full `SKILL.md`.

    Example input:
        list_skills()

    Example output:
        {"status": "success", "skills": [{"name": "docs-writer", "description": "..."}]}
    """
    discovered = _discover_skills()
    skills = [
        {
            "name": skill["name"],
            "description": skill["description"],
            "path": skill["path"],
            "directory": skill["directory"],
            "invalid": skill.get("invalid", False),
            "source": skill.get("source"),
        }
        for skill in discovered
    ]
    return {
        "status": "success",
        "skills_roots": [
            {"path": str(path), "source": source}
            for path, source in _candidate_skills_roots()
        ],
        "skills": skills,
    }


def activate_skill(skill_name: str) -> dict:
    """Load the full `SKILL.md` body for a discovered skill on demand.

    Example input:
        skill_name="docs-writer"

    Example output:
        {"status": "success", "activated_skill": {"name": "docs-writer", "body": "..."}}
    """
    for skill in _discover_skills():
        if skill["name"] != skill_name:
            continue
        if skill.get("invalid"):
            return {
                "status": "error",
                "error_message": f"Skill is invalid and cannot be activated: {skill_name}",
                "skill": skill,
            }
        return {
            "status": "success",
            "activated_skill": {
                "name": skill["name"],
                "description": skill["description"],
                "path": skill["path"],
                "directory": skill["directory"],
                "source": skill.get("source"),
                "resource_files": skill["resource_files"],
                "body": skill["body"],
            },
            "instruction": (
                "Prioritize the activated SKILL.md workflow while adapting it to "
                "the current repository and tool results."
            ),
        }

    return {
        "status": "error",
        "error_message": f"Unknown skill: {skill_name}",
        "available_skills": [skill["name"] for skill in _discover_skills()],
    }
