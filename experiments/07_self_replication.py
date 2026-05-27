from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from replication import build_exact_copy_program, offspring_matches_parent, run_replication_program


def main() -> None:
    genome = build_exact_copy_program()
    state = run_replication_program(genome)
    print(f"experiment: 07_self_replication")
    print(f"parent_len: {len(genome.codons)}")
    print(f"offspring_len: {len(state.offspring or ())}")
    print(f"replicated: {offspring_matches_parent(state)}")
    print("trace_examples:")
    for event in state.trace[:8]:
        print(f"  - step {event['step']}: {event['op']} {event['note']}")
    if not offspring_matches_parent(state):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
