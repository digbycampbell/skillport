"""Full-text search runs on a real LanceDB index, not the substring fallback."""

from pathlib import Path

from skillport.modules.indexing.internal.lancedb import IndexStore
from skillport.shared.config import Config


def _write_skill(root: Path, name: str, description: str, category: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n"
        f"metadata:\n  skillport:\n    category: {category}\n---\nBody\n",
        encoding="utf-8",
    )


def test_index_build_creates_a_usable_fts_index(tmp_path, capsys):
    skills = tmp_path / "skills"
    _write_skill(skills, "pdf-tools", "Extract text from PDF files", "office")
    _write_skill(skills, "deploy", "Ship a release to production", "ops")
    config = Config(
        skills_dir=skills,
        db_path=tmp_path / "db" / "skills.lancedb",
        embedding_provider="none",
    )
    store = IndexStore(config)
    store.initialize_index()

    assert "FTS index creation failed" not in capsys.readouterr().err
    by_description = store.search("production", limit=5)
    by_category = store.search("office", limit=5)
    assert [(r["id"], r["_source"]) for r in by_description] == [("deploy", "fts")]
    assert [(r["id"], r["_source"]) for r in by_category] == [("pdf-tools", "fts")]
