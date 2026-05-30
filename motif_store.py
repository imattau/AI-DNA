from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from genome import Motif


class MotifStore:
    """Thin SQLite wrapper for persisting and querying Motif objects."""

    def __init__(self, db_path: str | Path = "data/motifs.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS motifs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                task TEXT NOT NULL,
                experiment TEXT NOT NULL,
                generation INTEGER NOT NULL,
                gate_err REAL NOT NULL,
                lineage_id TEXT NOT NULL,
                codons TEXT NOT NULL,
                origin_signals TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def save(
        self,
        motif: Motif,
        role: str,
        task: str,
        gate_err: float,
        generation: int,
        experiment: str,
    ) -> None:
        """Insert one motif row."""
        self._conn.execute(
            "INSERT INTO motifs "
            "(role, task, experiment, generation, gate_err, lineage_id, codons, origin_signals) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                role,
                task,
                experiment,
                generation,
                gate_err,
                motif.origin_lineage,
                json.dumps(list(motif.pattern)),
                json.dumps(list(motif.origin_signals)),
            ),
        )
        self._conn.commit()

    def query(
        self,
        role: str | None = None,
        task: str | None = None,
        top_k: int = 4,
    ) -> list[Motif]:
        """Return up to top_k Motif objects ordered by gate_err ASC."""
        clauses: list[str] = []
        params: list[object] = []
        if role is not None:
            clauses.append("role = ?")
            params.append(role)
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT codons, origin_signals, lineage_id, task FROM motifs "
            f"{where} ORDER BY gate_err ASC LIMIT ?",
            params + [top_k],
        ).fetchall()
        result: list[Motif] = []
        for codons_json, signals_json, lineage_id, origin_task in rows:
            pattern = tuple(json.loads(codons_json))
            origin_signals = tuple(json.loads(signals_json))
            result.append(
                Motif(
                    pattern=pattern,
                    origin_lineage=lineage_id,
                    origin_task=origin_task,
                    origin_signals=origin_signals,
                )
            )
        return result

    def close(self) -> None:
        self._conn.close()
