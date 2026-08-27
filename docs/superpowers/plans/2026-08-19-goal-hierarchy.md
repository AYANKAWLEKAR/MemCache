# Goal Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Task nodes a strict-tree parent relation (`SUBGOAL_OF`), infer that relation from conversation with a precision-first shortlist + single-question Ollama placement call, and make retrieval pull context up the lineage — the `Current task:` line shows the path, Known Failures scopes to the whole chain, and the lineage Task nodes seed activation so upstream episodes and tool calls surface with provenance.

**Architecture:** Additive overlay on the existing Task tier. One new edge type, one new Task property (`last_advanced_at`), one new pure module (`task_hierarchy.py`) for shortlist + parse, one new worker step (`_write_hierarchy`) under the same "ingest never fails because inference failed" contract, and three retrieval call sites widened from "the active task" to "the active task's lineage". Structural pull comes from seeding the lineage `Task` nodes themselves (all-structural edges to episodes and tool calls, so the arithmetic clears the floor); topical pull stays as it is (leaf task entities at 0.6).

**Tech Stack:** Python 3.13, FastAPI, Neo4j 5 (Bolt, no GDS), PostgreSQL + SQLAlchemy 2.0 `text()`, Celery (eager in tests), spaCy, sentence-transformers MiniLM, Ollama `qwen2.5:3b`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-goal-hierarchy-design.md`

## Global Constraints

- **Precision beats coverage.** Every ambiguous placement resolves to *no edge*. A hallucinated id, a contradictory verdict, or a hierarchy invariant violation writes nothing and logs.
- **An ingest never fails because placement failed.** `_write_hierarchy` catches everything at its boundary. L1–L4 are unaffected.
- **Strict tree.** At most one outgoing `SUBGOAL_OF` per Task. Enforced in `TaskStore`, not by Neo4j.
- **Two timestamps.** `updated_at` bubbles to ancestors (candidate ordering); `last_advanced_at` is direct-only (active-task selection). Read `last_advanced_at` as `coalesce(t.last_advanced_at, t.updated_at)` everywhere so pre-existing Tasks need no migration.
- **Ancestor walks ignore `status`.** No cascade on close.
- **LLM judgement is a reported metric, never a gate, unless measurement earns it.** Deterministic tests mock the adjudicator and the placement call; the agentic tests gate on what the deterministic path already proves and report the rest under `[metric]`.
- Node ids in neighborhoods are `Label:key` strings; a Task's is `Task:<uuid>`.
- Run tests with `.venv/bin/python -m pytest`. Stack must be up: `docker compose up -d redis postgres neo4j`. Ollama must serve `qwen2.5:3b` at `http://localhost:11434` for agentic tests.
- Commit after every task with the `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer. `*.md` is gitignored — use `git add -f` for docs.
- The full suite (243 tests) must stay green throughout. Run `.venv/bin/python -m pytest -m "not agentic" -q` after each task; run the agentic set in Task 10.

## File Structure

| File | Responsibility |
|------|----------------|
| `app/config.py` (modify) | Four knobs: `task_max_depth`, `task_placement_candidates`, `task_placement_min_score`, `proactive_task_node_seed`, `proactive_task_depth_decay`. |
| `app/services/activation.py` (modify) | `SUBGOAL_OF` prior in `EDGE_PRIORS`. |
| `app/services/task_store.py` (modify) | `TaskHierarchyError`, `PlacementCandidate`, `last_advanced_at`, `set_parent`, walks, `active_task`, `list_placement_candidates`, `task_evidence`, bubbling in `link_episode`. |
| `app/services/task_hierarchy.py` (new) | Pure: `PlacementVerdict`, `parse_placement`, `build_placement_prompt`, `shortlist_candidates`; I/O: `adjudicate_placement`. |
| `app/services/workbench_store.py` (modify) | `failed_calls(task_ids=...)` lineage scope + rank. |
| `app/services/neo4j_store.py` (modify) | `fetch_neighborhood(task_ids=...)` start points. |
| `app/services/proactive.py` (modify) | `build_seeds(task_nodes=...)`. |
| `app/services/retrieval.py` (modify) | Three call sites → `active_task` + lineage; `Current task:` path line; lineage Task seeds. |
| `app/workers/tasks.py` (modify) | `_write_hierarchy` between `_write_task` and `_claim_workbench`. |
| `tests/test_activation.py` (modify) | `SUBGOAL_OF` weight. |
| `tests/test_task_store.py` (modify) | Hierarchy store tests. |
| `tests/test_task_hierarchy.py` (new) | Pure parse/prompt/shortlist tests. |
| `tests/test_task_worker.py` (modify) | Placement wiring with both calls mocked. |
| `tests/test_workbench_store.py` (modify) | Lineage-scoped `failed_calls`. |
| `tests/test_proactive_assembly.py` (modify) | `task_nodes` seeds. |
| `tests/test_l3_neo4j.py` or new `tests/test_neighborhood_task_seeds.py` | `fetch_neighborhood` from Task ids. |
| `tests/test_goal_hierarchy_retrieval.py` (new) | Hand-built graph: path line, Known Failures lineage, parent ToolCall via Task seed, severed tests. |
| `tests/agentic/hierarchy_scenarios.py` (new) | Four 3-session scenarios. |
| `tests/agentic/test_goal_hierarchy.py` (new) | Real-Ollama runs with `[context]`/`[tree]`/`[metric]` output. |
| `steps taken/l4_obstacles_and_decisions.md` (modify) | Calibration + gate-promotion addendum. |
| `README.md` (modify) | Graph diagram + one paragraph. |

---

### Task 1: Config knobs and the `SUBGOAL_OF` prior

**Files:**
- Modify: `app/config.py:84` (after `task_candidate_limit`) and `:102-106` (proactive block)
- Modify: `app/services/activation.py:22-33` (`EDGE_PRIORS`)
- Test: `tests/test_activation.py`

**Interfaces:**
- Produces: `settings.task_max_depth: int = 8`, `settings.task_placement_candidates: int = 3`, `settings.task_placement_min_score: float = 0.0`, `settings.proactive_task_node_seed: float = 0.2`, `settings.proactive_task_depth_decay: float = 0.7`, `EDGE_PRIORS["SUBGOAL_OF"] == 0.9`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_activation.py`:

```python
def test_subgoal_of_is_a_structural_edge_at_full_prior():
    """SUBGOAL_OF joins the tree-structure class (PURSUES/ADVANCES): uncounted,
    0.9, so activation crosses from a Task to its parent at the same strength
    it crosses from an Episode to its Task."""
    from app.services.activation import EDGE_PRIORS, COUNTED_EDGES, edge_weight

    assert EDGE_PRIORS["SUBGOAL_OF"] == pytest.approx(0.9)
    assert "SUBGOAL_OF" not in COUNTED_EDGES
    assert edge_weight("SUBGOAL_OF", 1) == pytest.approx(0.9)
    assert edge_weight("SUBGOAL_OF", 50) == pytest.approx(0.9)


def test_hierarchy_config_defaults():
    from app.config import settings

    assert settings.task_max_depth == 8
    assert settings.task_placement_candidates == 3
    assert settings.task_placement_min_score == 0.0
    assert settings.proactive_task_node_seed == pytest.approx(0.2)
    assert settings.proactive_task_depth_decay == pytest.approx(0.7)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_activation.py -k "subgoal_of or hierarchy_config" -v`
Expected: 2 FAIL (KeyError `SUBGOAL_OF`; AttributeError `task_max_depth`).

- [ ] **Step 3: Implement**

In `app/config.py`, after `task_candidate_limit: int = 20`:

```python
    # Goal hierarchy (SUBGOAL_OF). Depth cap bounds ancestor/descendant walks —
    # a fetch-size guard, not a semantic limit. Placement candidates are the
    # shortlist handed to the placement call; the min score is off by default
    # because with ≤3 candidates the model is the precision gate (the
    # unrelated_stays_flat agentic scenario measures it) — it is the lever if
    # measurement shows over-linking.
    task_max_depth: int = 8
    task_placement_candidates: int = 3
    task_placement_min_score: float = 0.0
```

In the proactive block, after `proactive_task_seed: float = 0.6`:

```python
    # Lineage Task nodes seed activation directly (structural pull up the
    # tree). Task->ADVANCES->Episode->INVOKED->ToolCall is all-structural
    # (0.9 each), so at 0.2·0.7^d the leaf's tool calls land at 0.104, the
    # parent's at 0.073, the grandparent's at 0.051, depth 3 dies. Live seeds
    # still win (1.0·MENTIONS(1)·0.8 = 0.164 > 0.144). Spec §4c; pinned by a
    # deterministic test; re-measured with calibrate_activation.py.
    proactive_task_node_seed: float = 0.2
    proactive_task_depth_decay: float = 0.7
```

In `app/services/activation.py` `EDGE_PRIORS`, after `"ADVANCES": 1.0,`:

```python
    "SUBGOAL_OF": 0.9,
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_activation.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/services/activation.py tests/test_activation.py
git commit -m "feat: hierarchy config knobs and SUBGOAL_OF activation prior

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `TaskStore` — `set_parent`, walks, invariants

**Files:**
- Modify: `app/services/task_store.py`
- Test: `tests/test_task_store.py`

**Interfaces:**
- Produces: `TaskHierarchyError(RuntimeError)`, `TaskStore.set_parent(child_id: str, parent_id: str) -> None`, `TaskStore.get_parent(task_id) -> TaskRow | None`, `TaskStore.get_ancestors(task_id) -> list[TaskRow]` (nearest first), `TaskStore.get_children(task_id) -> list[TaskRow]`, `TaskStore.get_descendant_ids(task_id) -> set[str]`, `TaskStore.get_lineage_ids(task_id) -> list[str]` (`[task_id, parent, …]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_task_store.py`:

```python
# ---------------------------------------------------------------- hierarchy


def test_set_parent_creates_subgoal_edge_and_is_idempotent(store, driver, user_id):
    _seed_profile(driver, user_id)
    parent = store.create_task(user_id, "Ship telemetry v2")
    child = store.create_task(user_id, "Migrate telemetry schema")

    store.set_parent(child, parent)
    store.set_parent(child, parent)  # same edge again: no-op

    with driver.session() as s:
        n = s.run(
            "MATCH (:Task {id: $c})-[r:SUBGOAL_OF]->(:Task {id: $p}) RETURN count(r) AS n",
            c=child, p=parent,
        ).single()["n"]
    assert n == 1
    got = store.get_parent(child)
    assert got is not None and got.id == parent
    assert store.get_parent(parent) is None


def test_set_parent_refuses_a_second_different_parent(store, driver, user_id):
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    a = store.create_task(user_id, "A")
    b = store.create_task(user_id, "B")
    child = store.create_task(user_id, "child")
    store.set_parent(child, a)
    with pytest.raises(TaskHierarchyError):
        store.set_parent(child, b)
    assert store.get_parent(child).id == a  # unchanged


def test_set_parent_refuses_self(store, driver, user_id):
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    t = store.create_task(user_id, "loop")
    with pytest.raises(TaskHierarchyError):
        store.set_parent(t, t)


def test_set_parent_refuses_a_parent_inside_the_childs_subtree(store, driver, user_id):
    """root <- mid <- leaf; then root under leaf would be a cycle."""
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    mid = store.create_task(user_id, "mid")
    leaf = store.create_task(user_id, "leaf")
    store.set_parent(mid, root)
    store.set_parent(leaf, mid)
    with pytest.raises(TaskHierarchyError):
        store.set_parent(root, leaf)
    with pytest.raises(TaskHierarchyError):
        store.set_parent(root, mid)
    assert store.get_parent(root) is None


def test_set_parent_refuses_cross_user(store, driver, user_id):
    """Cross-user parentage would leak one person's failures into another's
    context — this must fail loudly, never degrade."""
    from app.services.task_store import TaskHierarchyError

    _seed_profile(driver, user_id)
    other = f"{user_id}-other"
    _seed_profile(driver, other)
    try:
        mine = store.create_task(user_id, "mine")
        theirs = store.create_task(other, "theirs")
        with pytest.raises(TaskHierarchyError):
            store.set_parent(mine, theirs)
        with pytest.raises(TaskHierarchyError):
            store.set_parent(theirs, mine)
    finally:
        with driver.session() as s:
            s.run(
                "MATCH (p:UserProfile {user_id: $uid}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) "
                "DETACH DELETE p, t",
                uid=other,
            )


def test_ancestors_children_descendants_and_lineage(store, driver, user_id):
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    mid = store.create_task(user_id, "mid")
    leaf = store.create_task(user_id, "leaf")
    sib = store.create_task(user_id, "sibling")
    store.set_parent(mid, root)
    store.set_parent(leaf, mid)
    store.set_parent(sib, root)

    assert [t.id for t in store.get_ancestors(leaf)] == [mid, root]  # nearest first
    assert store.get_ancestors(root) == []
    assert {t.id for t in store.get_children(root)} == {mid, sib}
    assert store.get_children(leaf) == []
    assert store.get_descendant_ids(root) == {mid, leaf, sib}
    assert store.get_descendant_ids(leaf) == set()
    assert store.get_lineage_ids(leaf) == [leaf, mid, root]
    assert store.get_lineage_ids(root) == [root]


def test_ancestor_walk_ignores_status(store, driver, user_id):
    """A closed parent with an open child is legitimate; the walk must not
    truncate at it."""
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    leaf = store.create_task(user_id, "leaf")
    store.set_parent(leaf, root)
    store.close_task(root)
    anc = store.get_ancestors(leaf)
    assert [t.id for t in anc] == [root]
    assert anc[0].status == "done"


def test_ancestor_walk_is_bounded_by_max_depth(store, driver, user_id, monkeypatch):
    from app.config import settings

    _seed_profile(driver, user_id)
    monkeypatch.setattr(settings, "task_max_depth", 3)
    ids = [store.create_task(user_id, f"t{i}") for i in range(6)]
    for child, parent in zip(ids[1:], ids[:-1]):
        store.set_parent(child, parent)  # t5 -> t4 -> ... -> t0
    anc = store.get_ancestors(ids[5])
    assert len(anc) == 3, "depth cap must bound the walk"
    assert [t.id for t in anc] == [ids[4], ids[3], ids[2]]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_task_store.py -k "parent or ancestor or lineage" -v`
Expected: FAIL — `AttributeError: 'TaskStore' object has no attribute 'set_parent'` / ImportError `TaskHierarchyError`.

- [ ] **Step 3: Implement**

In `app/services/task_store.py`, add after the imports:

```python
from app.config import settings


class TaskHierarchyError(RuntimeError):
    """A SUBGOAL_OF write would break the tree: second parent, self-loop,
    cycle, or cross-user parentage. Raised loudly — never degraded — because a
    wrong parent edge injects one goal's failures into another goal's context.
    """
```

Add these methods to `TaskStore` (after `close_task`):

```python
    # ------------------------------------------------------------ hierarchy

    def _depth_cap(self) -> int:
        # Cypher variable-length bounds must be literals; clamp to something sane.
        return max(1, min(int(settings.task_max_depth), 32))

    def get_parent(self, task_id: str) -> TaskRow | None:
        q = """
        MATCH (:Task {id: $task_id})-[:SUBGOAL_OF]->(p:Task)
        RETURN p.id AS id, p.title AS title, p.status AS status,
               p.created_at AS created_at, p.updated_at AS updated_at,
               p.last_advanced_at AS last_advanced_at
        """
        with self._driver.session() as session:
            rec = session.run(q, task_id=task_id).single()
        return _row(rec) if rec else None

    def get_ancestors(self, task_id: str) -> list[TaskRow]:
        """Nearest-first, root last; empty for a root. Bounded by `task_max_depth`
        as a fetch cap. Ignores status on purpose — a closed parent with an open
        child is legitimate and filtering would silently truncate the path."""
        d = self._depth_cap()
        q = f"""
        MATCH p = (:Task {{id: $task_id}})-[:SUBGOAL_OF*1..{d}]->(a:Task)
        WITH a, length(p) AS depth
        ORDER BY depth ASC
        RETURN a.id AS id, a.title AS title, a.status AS status,
               a.created_at AS created_at, a.updated_at AS updated_at,
               a.last_advanced_at AS last_advanced_at
        """
        with self._driver.session() as session:
            return [_row(r) for r in session.run(q, task_id=task_id)]

    def get_children(self, task_id: str) -> list[TaskRow]:
        q = """
        MATCH (c:Task)-[:SUBGOAL_OF]->(:Task {id: $task_id})
        RETURN c.id AS id, c.title AS title, c.status AS status,
               c.created_at AS created_at, c.updated_at AS updated_at,
               c.last_advanced_at AS last_advanced_at
        ORDER BY c.updated_at DESC
        """
        with self._driver.session() as session:
            return [_row(r) for r in session.run(q, task_id=task_id)]

    def get_descendant_ids(self, task_id: str) -> set[str]:
        d = self._depth_cap()
        q = f"""
        MATCH (c:Task)-[:SUBGOAL_OF*1..{d}]->(:Task {{id: $task_id}})
        RETURN DISTINCT c.id AS id
        """
        with self._driver.session() as session:
            return {r["id"] for r in session.run(q, task_id=task_id)}

    def get_lineage_ids(self, task_id: str) -> list[str]:
        """`[task_id, parent, grandparent, ...]` — what retrieval scopes by."""
        return [task_id] + [a.id for a in self.get_ancestors(task_id)]

    def set_parent(self, child_id: str, parent_id: str) -> None:
        """MERGE `(child)-[:SUBGOAL_OF]->(parent)`; no-op if that edge exists.

        Raises `TaskHierarchyError` for: self, a different existing parent, a
        parent inside the child's own subtree (cycle), or a parent pursued by a
        different profile. All checks run before any write.
        """
        if child_id == parent_id:
            raise TaskHierarchyError(f"task {child_id} cannot be its own parent")
        current = self.get_parent(child_id)
        if current is not None:
            if current.id == parent_id:
                return
            raise TaskHierarchyError(
                f"task {child_id} already has parent {current.id}; refusing {parent_id}"
            )
        if parent_id in self.get_descendant_ids(child_id):
            raise TaskHierarchyError(
                f"task {parent_id} is a descendant of {child_id}; parenting would cycle"
            )
        q_owner = """
        MATCH (c:Task {id: $child}), (p:Task {id: $parent})
        OPTIONAL MATCH (uc:UserProfile)-[:PURSUES]->(c)
        OPTIONAL MATCH (up:UserProfile)-[:PURSUES]->(p)
        RETURN uc.user_id AS child_owner, up.user_id AS parent_owner
        """
        with self._driver.session() as session:
            rec = session.run(q_owner, child=child_id, parent=parent_id).single()
            if rec is None:
                raise TaskHierarchyError(f"task {child_id} or {parent_id} does not exist")
            if rec["child_owner"] != rec["parent_owner"] or rec["child_owner"] is None:
                raise TaskHierarchyError(
                    f"tasks {child_id} ({rec['child_owner']!r}) and {parent_id} "
                    f"({rec['parent_owner']!r}) are not pursued by the same profile"
                )
            session.run(
                """
                MATCH (c:Task {id: $child}), (p:Task {id: $parent})
                MERGE (c)-[:SUBGOAL_OF]->(p)
                """,
                child=child_id,
                parent=parent_id,
            )
```

Add a module-level row helper (above the class) and make `TaskRow` carry the new field with a default so existing constructions keep working:

```python
@dataclass(frozen=True)
class TaskRow:
    """One Task node as seen by callers."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    last_advanced_at: str | None = None


def _row(record) -> TaskRow:
    return TaskRow(
        id=record["id"],
        title=record["title"],
        status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        last_advanced_at=record.get("last_advanced_at") if hasattr(record, "get") else record["last_advanced_at"],
    )
```

(Neo4j `Record` supports `.get`; keep the `hasattr` guard so a plain dict also works in unit tests.)

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_task_store.py -v`
Expected: all PASS (existing 7 + new 8).

- [ ] **Step 5: Commit**

```bash
git add app/services/task_store.py tests/test_task_store.py
git commit -m "feat: SUBGOAL_OF tree in TaskStore — set_parent, walks, invariants

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Two timestamps — bubbling `updated_at`, direct-only `last_advanced_at`, `active_task`

**Files:**
- Modify: `app/services/task_store.py` (`create_task`, `get_task`, `list_open_tasks`, `link_episode`; new `active_task`)
- Test: `tests/test_task_store.py`

**Interfaces:**
- Produces: `TaskStore.active_task(user_id: str) -> TaskRow | None`; `link_episode` bubbles `updated_at`; `TaskRow.last_advanced_at` populated by every read.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_task_store.py`:

```python
# ------------------------------------------------------- two timestamps


def test_link_episode_bubbles_updated_at_but_not_last_advanced_at(store, driver, user_id):
    """Advancing a leaf must keep its ancestors 'recently active' for the
    adjudicator (updated_at) WITHOUT making them look like the active task
    (last_advanced_at) — that split is the whole reason there are two fields."""
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    leaf = store.create_task(user_id, "leaf")
    store.set_parent(leaf, root)
    before = store.get_task(root)

    store.link_episode(leaf, episode_id=-930010)

    after_root = store.get_task(root)
    after_leaf = store.get_task(leaf)
    assert after_root.updated_at > before.updated_at, "bubbling did not touch the ancestor"
    assert after_root.last_advanced_at == before.last_advanced_at, (
        "ancestor's last_advanced_at moved — it must be direct-only"
    )
    assert after_leaf.last_advanced_at > before.updated_at


def test_active_task_is_the_leaf_after_bubbling(store, driver, user_id):
    """The exact tie the split prevents: after bubbling, root and leaf share
    updated_at; active_task must still choose the leaf, deterministically."""
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    leaf = store.create_task(user_id, "leaf")
    store.set_parent(leaf, root)
    store.link_episode(leaf, episode_id=-930011)

    for _ in range(5):  # would flap if the tie were resolved by ORDER BY updated_at
        active = store.active_task(user_id)
        assert active is not None and active.id == leaf


def test_active_task_follows_direct_advancement_not_bubbling(store, driver, user_id):
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    a = store.create_task(user_id, "a")
    b = store.create_task(user_id, "b")
    store.set_parent(a, root)
    store.set_parent(b, root)
    store.link_episode(a, episode_id=-930012)
    assert store.active_task(user_id).id == a
    store.link_episode(b, episode_id=-930013)
    assert store.active_task(user_id).id == b
    # Advancing the root directly makes the root active.
    store.link_episode(root, episode_id=-930014)
    assert store.active_task(user_id).id == root


def test_active_task_ignores_done_and_returns_none_when_empty(store, driver, user_id):
    _seed_profile(driver, user_id)
    assert store.active_task(user_id) is None
    t = store.create_task(user_id, "t")
    assert store.active_task(user_id).id == t
    store.close_task(t)
    assert store.active_task(user_id) is None


def test_active_task_backfills_from_updated_at_for_legacy_rows(store, driver, user_id):
    """Pre-hierarchy Tasks have no last_advanced_at; they must still be selectable."""
    _seed_profile(driver, user_id)
    t = store.create_task(user_id, "legacy")
    with driver.session() as s:
        s.run("MATCH (t:Task {id: $id}) REMOVE t.last_advanced_at", id=t)
    assert store.active_task(user_id).id == t
    assert store.get_task(t).last_advanced_at is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_task_store.py -k "bubbles or active_task" -v`
Expected: FAIL (`active_task` missing; bubbling assertions fail).

- [ ] **Step 3: Implement**

In `task_store.py`:

`create_task` — add `last_advanced_at: $now` to the CREATE properties:

```python
        CREATE (t:Task {
            id: $task_id,
            title: $title,
            status: 'open',
            created_at: $now,
            updated_at: $now,
            last_advanced_at: $now
        })
```

`get_task` and `list_open_tasks` — add `t.last_advanced_at AS last_advanced_at` to RETURN and build rows with `_row(record)`.

`link_episode` — replace the query:

```python
    def link_episode(self, task_id: str, *, episode_id: int) -> None:
        """MERGE `(:Episode)-[:ADVANCES]->(:Task)`; bump the target's two
        timestamps; bubble `updated_at` (only) to every ancestor.

        One statement so the direct bump and the bubbling are atomic. The
        ancestor bound is the depth cap, a fetch guard — a real tree is never
        that deep, and if it were, the far ancestors are the ones that can
        afford to age out.
        """
        d = self._depth_cap()
        q = f"""
        MATCH (t:Task {{id: $task_id}})
        MERGE (e:Episode {{id: $episode_id}})
        MERGE (e)-[:ADVANCES]->(t)
        SET t.updated_at = $now, t.last_advanced_at = $now
        WITH t
        OPTIONAL MATCH (t)-[:SUBGOAL_OF*1..{d}]->(a:Task)
        SET a.updated_at = $now
        """
        with self._driver.session() as session:
            session.run(q, task_id=task_id, episode_id=episode_id, now=_now())
```

Add `active_task`:

```python
    def active_task(self, user_id: str) -> TaskRow | None:
        """The open Task most recently *directly* advanced — the leaf being
        worked, never a parent that merely bubbled. Legacy rows without
        `last_advanced_at` fall back to `updated_at`."""
        q = """
        MATCH (:UserProfile {user_id: $user_id})-[:PURSUES]->(t:Task {status: 'open'})
        RETURN t.id AS id, t.title AS title, t.status AS status,
               t.created_at AS created_at, t.updated_at AS updated_at,
               t.last_advanced_at AS last_advanced_at
        ORDER BY coalesce(t.last_advanced_at, t.updated_at) DESC, t.created_at DESC
        LIMIT 1
        """
        with self._driver.session() as session:
            rec = session.run(q, user_id=user_id).single()
        return _row(rec) if rec else None
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_task_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/task_store.py tests/test_task_store.py
git commit -m "feat: last_advanced_at + bubbling updated_at; active_task picks the leaf

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Placement candidates and evidence (`TaskStore`)

**Files:**
- Modify: `app/services/task_store.py`
- Test: `tests/test_task_store.py`

**Interfaces:**
- Produces: `PlacementCandidate(id: str, title: str, is_root: bool, updated_at: str, entities: frozenset[str], sessions: frozenset[str])`, `TaskStore.list_placement_candidates(user_id, *, subject_id, limit) -> list[PlacementCandidate]`, `TaskStore.task_evidence(task_id) -> PlacementCandidate | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_task_store.py`:

```python
# ------------------------------------------------- placement candidates


def _episode_with(driver, eid: int, session_id: str, entities: list[str]) -> None:
    with driver.session() as s:
        s.run(
            """
            MERGE (se:Session {id: $sid})
            MERGE (e:Episode {id: $eid}) SET e.session_id = $sid
            MERGE (se)-[:HAS_EPISODE]->(e)
            WITH e
            UNWIND $names AS n
            MERGE (ent:Entity {name: n})
            MERGE (e)-[:MENTIONS]->(ent)
            """,
            sid=session_id, eid=eid, names=entities,
        )


def _drop_episode(driver, eid: int, session_id: str) -> None:
    with driver.session() as s:
        s.run("MATCH (e:Episode {id: $eid}) DETACH DELETE e", eid=eid)
        s.run("MATCH (se:Session {id: $sid}) DETACH DELETE se", sid=session_id)


def test_placement_candidates_carry_evidence_and_exclude_subject_subtree(store, driver, user_id):
    _seed_profile(driver, user_id)
    root = store.create_task(user_id, "root")
    child = store.create_task(user_id, "child")
    other = store.create_task(user_id, "other")
    done = store.create_task(user_id, "done")
    store.set_parent(child, root)
    store.close_task(done)
    sid = f"{user_id}-s"
    try:
        _episode_with(driver, -930020, sid, ["clickhouse", "alembic"])
        store.link_episode(other, episode_id=-930020)

        cands = store.list_placement_candidates(user_id, subject_id=root, limit=20)
        ids = {c.id for c in cands}
        assert root not in ids, "subject must be excluded"
        assert child not in ids, "subject's descendants must be excluded"
        assert done not in ids, "closed tasks are not candidates"
        assert ids == {other}
        (o,) = cands
        assert o.is_root is True
        assert o.entities == frozenset({"clickhouse", "alembic"})
        assert o.sessions == frozenset({sid})

        # From the child's point of view, root and other are both candidates.
        by_child = {c.id: c for c in store.list_placement_candidates(user_id, subject_id=child, limit=20)}
        assert set(by_child) == {root, other}
        assert by_child[root].is_root is True

        # is_root is false for a parented task seen as a candidate.
        by_other = {c.id: c for c in store.list_placement_candidates(user_id, subject_id=other, limit=20)}
        assert by_other[child].is_root is False
    finally:
        _drop_episode(driver, -930020, sid)


def test_placement_candidates_respect_limit_and_recency(store, driver, user_id):
    _seed_profile(driver, user_id)
    a = store.create_task(user_id, "a")
    b = store.create_task(user_id, "b")
    subj = store.create_task(user_id, "subject")
    store.link_episode(a, episode_id=-930021)  # a is now most recent
    try:
        cands = store.list_placement_candidates(user_id, subject_id=subj, limit=1)
        assert [c.id for c in cands] == [a]
    finally:
        with driver.session() as s:
            s.run("MATCH (e:Episode {id: -930021}) DETACH DELETE e")


def test_task_evidence_for_subject(store, driver, user_id):
    _seed_profile(driver, user_id)
    t = store.create_task(user_id, "subject")
    sid = f"{user_id}-ev"
    try:
        _episode_with(driver, -930022, sid, ["kafka"])
        store.link_episode(t, episode_id=-930022)
        ev = store.task_evidence(t)
        assert ev is not None
        assert ev.id == t and ev.is_root is True
        assert ev.entities == frozenset({"kafka"}) and ev.sessions == frozenset({sid})
        assert store.task_evidence("no-such") is None
    finally:
        _drop_episode(driver, -930022, sid)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_task_store.py -k "placement or evidence" -v`
Expected: FAIL (`list_placement_candidates` missing).

- [ ] **Step 3: Implement**

In `task_store.py`, add the dataclass after `TaskRow`:

```python
@dataclass(frozen=True)
class PlacementCandidate:
    """A Task plus the graph evidence the placement shortlist scores."""

    id: str
    title: str
    is_root: bool
    updated_at: str
    entities: frozenset[str]
    sessions: frozenset[str]
```

Add to `TaskStore`:

```python
    # ------------------------------------------------ placement evidence

    _EVIDENCE_RETURN = """
        RETURN t.id AS id, t.title AS title, t.updated_at AS updated_at,
               NOT exists((t)-[:SUBGOAL_OF]->()) AS is_root,
               [x IN collect(DISTINCT ent.name) WHERE x IS NOT NULL] AS entities,
               [x IN collect(DISTINCT se.id) WHERE x IS NOT NULL] AS sessions
    """

    def task_evidence(self, task_id: str) -> PlacementCandidate | None:
        q = f"""
        MATCH (t:Task {{id: $task_id}})
        OPTIONAL MATCH (t)<-[:ADVANCES]-(e:Episode)
        OPTIONAL MATCH (e)-[:MENTIONS]->(ent:Entity)
        OPTIONAL MATCH (e)<-[:HAS_EPISODE]-(se:Session)
        {self._EVIDENCE_RETURN}
        """
        with self._driver.session() as session:
            rec = session.run(q, task_id=task_id).single()
        return _candidate(rec) if rec else None

    def list_placement_candidates(
        self, user_id: str, *, subject_id: str, limit: int
    ) -> list[PlacementCandidate]:
        """Open Tasks this user pursues, minus the subject and its subtree,
        most recently active first, capped. One round-trip.

        The exclusion is what makes `child_of` cycle-safe by construction —
        nothing the model can name is inside the subject's own subtree.
        """
        excluded = self.get_descendant_ids(subject_id) | {subject_id}
        q = f"""
        MATCH (:UserProfile {{user_id: $user_id}})-[:PURSUES]->(t:Task {{status: 'open'}})
        WHERE NOT t.id IN $excluded
        WITH t ORDER BY t.updated_at DESC LIMIT $limit
        OPTIONAL MATCH (t)<-[:ADVANCES]-(e:Episode)
        OPTIONAL MATCH (e)-[:MENTIONS]->(ent:Entity)
        OPTIONAL MATCH (e)<-[:HAS_EPISODE]-(se:Session)
        {self._EVIDENCE_RETURN}
        ORDER BY t.updated_at DESC
        """
        with self._driver.session() as session:
            return [
                _candidate(r)
                for r in session.run(
                    q, user_id=user_id, excluded=list(excluded), limit=max(1, int(limit))
                )
            ]
```

And a module-level helper next to `_row`:

```python
def _candidate(record) -> PlacementCandidate:
    return PlacementCandidate(
        id=record["id"],
        title=record["title"],
        is_root=bool(record["is_root"]),
        updated_at=record["updated_at"],
        entities=frozenset(record["entities"] or []),
        sessions=frozenset(record["sessions"] or []),
    )
```

Note: `exists((t)-[:SUBGOAL_OF]->())` is the Neo4j 5 pattern-predicate form. If the deployed server rejects it, use `NOT (t)-[:SUBGOAL_OF]->()` inside a `WITH t, NOT (t)-[:SUBGOAL_OF]->() AS is_root` clause instead. Verify against the live server in Step 4.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_task_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/task_store.py tests/test_task_store.py
git commit -m "feat: placement candidates with graph evidence (entities, sessions, is_root)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `task_hierarchy.py` — pure parse, prompt, shortlist

**Files:**
- Create: `app/services/task_hierarchy.py`
- Create: `tests/test_task_hierarchy.py`

**Interfaces:**
- Consumes: `PlacementCandidate` from Task 4.
- Produces: `PlacementVerdict(relation: Literal["child_of","parent_of"], task_id: str)`, `parse_placement(text: str, valid_ids: set[str]) -> PlacementVerdict | None`, `build_placement_prompt(subject_title: str, candidates: list[tuple[str, str]]) -> str`, `shortlist_candidates(subject: PlacementCandidate, candidates: list[PlacementCandidate], *, similarity: Callable[[str, str], float], limit: int, min_score: float) -> list[PlacementCandidate]`, `placement_score(subject, cand, similarity) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_task_hierarchy.py`:

```python
"""Placement adjudication: parsing, prompt, and structural shortlist.

Pure functions — no Ollama, no DB. The parser is the safety boundary between a
3B model's output and the tree: everything malformed or contradictory must
degrade to "no edge", never to an exception and never to a wrong edge.
"""

from __future__ import annotations

import pytest

from app.services.task_hierarchy import (
    PlacementVerdict,
    build_placement_prompt,
    parse_placement,
    placement_score,
    shortlist_candidates,
)
from app.services.task_store import PlacementCandidate

A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
VALID = {A, B}


# ------------------------------------------------------------- parsing


def test_parses_child_of():
    v = parse_placement(f'{{"relation": "child_of", "task_id": "{A}"}}', VALID)
    assert v == PlacementVerdict(relation="child_of", task_id=A)


def test_parses_parent_of_in_fences_and_prose():
    text = f'Sure:\n```json\n{{"relation": "parent_of", "task_id": "{B}"}}\n```\nDone.'
    v = parse_placement(text, VALID)
    assert v == PlacementVerdict(relation="parent_of", task_id=B)


def test_none_relation_is_no_verdict():
    assert parse_placement('{"relation": "none", "task_id": null}', VALID) is None


def test_none_with_an_id_is_a_contradiction_and_degrades():
    assert parse_placement(f'{{"relation": "none", "task_id": "{A}"}}', VALID) is None


def test_relation_without_id_degrades():
    assert parse_placement('{"relation": "child_of", "task_id": null}', VALID) is None


def test_hallucinated_id_degrades():
    assert parse_placement('{"relation": "child_of", "task_id": "nope"}', VALID) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think it is a subgoal.",
        '{"relation": "child_of"',
        '{"task_id": "%s"}' % A,
        '{"relation": "sibling_of", "task_id": "%s"}' % A,
        '{"relation": 3, "task_id": "%s"}' % A,
        '{"relation": "child_of", "task_id": 42}',
        "[]",
    ],
)
def test_malformed_degrades_to_none(text):
    assert parse_placement(text, VALID) is None


# -------------------------------------------------------------- prompt


def test_prompt_lists_candidates_and_the_vocabulary():
    p = build_placement_prompt("Fix duplicate column", [(A, "Migrate schema"), (B, "Ship v2")])
    assert "Fix duplicate column" in p
    assert f"id: {A} | title: Migrate schema" in p
    assert f"id: {B} | title: Ship v2" in p
    for word in ("child_of", "parent_of", "none", "When unsure, answer none"):
        assert word in p


# ----------------------------------------------------------- shortlist


def _c(id_, title, *, ents=(), sess=(), root=True, updated="2026-08-19T00:00:00+00:00"):
    return PlacementCandidate(
        id=id_, title=title, is_root=root, updated_at=updated,
        entities=frozenset(ents), sessions=frozenset(sess),
    )


def _sim_table(table):
    def sim(a, b):
        return table.get((a, b), table.get((b, a), 0.0))
    return sim


def test_score_formula():
    subj = _c("s", "S", ents={"x", "y"}, sess={"s1"})
    cand = _c("c", "C", ents={"y", "z"}, sess={"s1", "s9"})
    sim = _sim_table({("S", "C"): 0.5})
    # 0.5*0.5 + 0.3*(1/3) + 0.2*1
    assert placement_score(subj, cand, sim) == pytest.approx(0.25 + 0.1 + 0.2)


def test_score_without_any_overlap_is_zero():
    subj = _c("s", "S", ents={"x"}, sess={"s1"})
    cand = _c("c", "C", ents={"z"}, sess={"s2"})
    assert placement_score(subj, cand, lambda a, b: 0.0) == 0.0


def test_shortlist_ranks_by_score_then_recency_and_caps():
    subj = _c("s", "S", ents={"x"}, sess={"s1"})
    hi = _c("hi", "HI", ents={"x"}, updated="2026-01-01T00:00:00+00:00")
    new = _c("new", "NEW", updated="2026-08-01T00:00:00+00:00")
    old = _c("old", "OLD", updated="2025-01-01T00:00:00+00:00")
    out = shortlist_candidates(subj, [old, new, hi], similarity=lambda a, b: 0.0, limit=2, min_score=0.0)
    assert [c.id for c in out] == ["hi", "new"]  # score, then newer first


def test_shortlist_min_score_cuts():
    subj = _c("s", "S", ents={"x"})
    weak = _c("w", "W")
    out = shortlist_candidates(subj, [weak], similarity=lambda a, b: 0.0, limit=3, min_score=0.05)
    assert out == []


def test_shortlist_excludes_subject_defensively_and_handles_empty():
    subj = _c("s", "S")
    assert shortlist_candidates(subj, [], similarity=lambda a, b: 1.0, limit=3, min_score=0.0) == []
    assert shortlist_candidates(subj, [subj], similarity=lambda a, b: 1.0, limit=3, min_score=0.0) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_task_hierarchy.py -v`
Expected: FAIL at import (`ModuleNotFoundError: app.services.task_hierarchy`).

- [ ] **Step 3: Implement**

Create `app/services/task_hierarchy.py`:

```python
"""Goal-hierarchy placement: shortlist by graph evidence, ask one question,
parse defensively.

A separate call from task adjudication, on purpose. Adjudication already asks a
3B model for goal extraction, same-or-new, and completion against up to twenty
candidates; bolting "and is it a subgoal of one of them" onto that call would
overload exactly the judgement this tier depends on. Here the model sees at
most three candidates and answers one question.

The parser is the safety boundary. A hallucinated id, a `none` that names an
id, a relation with no id, an unknown relation — all degrade to ``None`` ("no
edge"). A wrong SUBGOAL_OF edge injects one goal's failures into another goal's
context, so every ambiguous case resolves to nothing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Literal

import httpx

from app.config import Settings, settings as default_settings
from app.services.task_store import PlacementCandidate

logger = logging.getLogger(__name__)

Relation = Literal["child_of", "parent_of"]


@dataclass(frozen=True)
class PlacementVerdict:
    """A parsed, non-null placement: the subject is `relation` the named task."""

    relation: Relation
    task_id: str


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_RELATIONS = {"child_of", "parent_of", "none"}


def parse_placement(text: str, valid_task_ids: set[str]) -> PlacementVerdict | None:
    """Parse a model response into a verdict, or ``None`` when unusable or `none`."""
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not {"relation", "task_id"} <= set(data):
        return None
    relation, task_id = data["relation"], data["task_id"]
    if not isinstance(relation, str) or relation not in _RELATIONS:
        return None
    if task_id is not None and not isinstance(task_id, str):
        return None
    if relation == "none":
        if task_id is not None:
            logger.warning("placement said none but named %r; discarding", task_id)
        return None
    if task_id is None:
        return None
    if task_id not in valid_task_ids:
        logger.warning("placement referenced unknown task id %r; discarding", task_id)
        return None
    return PlacementVerdict(relation=relation, task_id=task_id)  # type: ignore[arg-type]


def build_placement_prompt(subject_title: str, candidates: list[tuple[str, str]]) -> str:
    """One question, fixed vocabulary, ≤3 candidates (capped by the caller)."""
    lines = "\n".join(f"- id: {tid} | title: {title}" for tid, title in candidates)
    return (
        "You organize a user's goals into a tree of goals and subgoals.\n"
        "Respond with ONLY a JSON object, no prose, no code fences.\n\n"
        f"New goal:\n{subject_title}\n\n"
        f"Existing goals:\n{lines}\n\n"
        "Decide how the new goal relates to ONE of the existing goals:\n"
        "child_of  = the new goal is a smaller step toward that existing goal.\n"
        "parent_of = that existing goal is a smaller step toward the new goal.\n"
        "none      = unrelated, siblings under some larger goal, or the same goal.\n"
        "When unsure, answer none.\n\n"
        "Respond with exactly:\n"
        '{"relation": <"child_of" | "parent_of" | "none">, '
        '"task_id": <the id of the existing goal the relation is with, or null>}'
    )


# ---------------------------------------------------------------- shortlist


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def placement_score(
    subject: PlacementCandidate,
    cand: PlacementCandidate,
    similarity: Callable[[str, str], float],
) -> float:
    """`0.5·title_similarity + 0.3·entity_jaccard + 0.2·[shares a session]`.

    Every term is symmetric — it says "these goals share territory," never
    which is larger. Direction is the model's job. Recency is deliberately not
    a term: it is not evidence of relatedness, and it would push every recent
    task past any min-score cut.
    """
    sim = max(0.0, min(1.0, float(similarity(subject.title, cand.title))))
    return (
        0.5 * sim
        + 0.3 * _jaccard(subject.entities, cand.entities)
        + 0.2 * (1.0 if subject.sessions & cand.sessions else 0.0)
    )


def shortlist_candidates(
    subject: PlacementCandidate,
    candidates: list[PlacementCandidate],
    *,
    similarity: Callable[[str, str], float],
    limit: int,
    min_score: float,
) -> list[PlacementCandidate]:
    """Top-`limit` candidates by score (recency breaks ties), above `min_score`."""
    scored = [
        (placement_score(subject, c, similarity), c.updated_at, c)
        for c in candidates
        if c.id != subject.id
    ]
    scored = [t for t in scored if t[0] >= min_score]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _, _, c in scored[: max(0, int(limit))]]


# ------------------------------------------------------------------ Ollama


def adjudicate_placement(
    subject_title: str,
    candidates: list[tuple[str, str]],
    *,
    settings: Settings | None = None,
    timeout_seconds: float = 60.0,
) -> PlacementVerdict | None:
    """Ask the model; parse; ``None`` on any failure. Same shape as
    `task_inference.adjudicate_task`."""
    cfg = settings or default_settings
    if not subject_title.strip() or not candidates:
        return None
    prompt = build_placement_prompt(subject_title, candidates)
    url = cfg.ollama_base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if (cfg.ollama_api_key or "").strip():
        headers["Authorization"] = f"Bearer {cfg.ollama_api_key.strip()}"
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("placement adjudication call failed: %s", exc)
        return None
    return parse_placement(text, {tid for tid, _ in candidates})
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_task_hierarchy.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/task_hierarchy.py tests/test_task_hierarchy.py
git commit -m "feat: placement adjudication — defensive parse, prompt, structural shortlist

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Worker wiring — `_write_hierarchy`

**Files:**
- Modify: `app/workers/tasks.py` (imports; new `_write_hierarchy`; two call sites after `_write_task`)
- Test: `tests/test_task_worker.py`

**Interfaces:**
- Consumes: `TaskStore.get_parent/list_placement_candidates/task_evidence/set_parent`, `shortlist_candidates`, `adjudicate_placement`, `_worker_embedder`.
- Produces: `_write_hierarchy(neo_driver, *, user_id, task_id, embedder) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_task_worker.py`:

```python
# ------------------------------------------------------------ hierarchy


def _mock_adjudicator(monkeypatch, goal, matches=None):
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_task",
        lambda summary, open_tasks, settings=None: TaskAdjudication(
            goal=goal, matches_task_id=matches, task_complete=False
        ),
    )


def _mock_placement(monkeypatch, fn):
    """fn(subject_title, candidates) -> PlacementVerdict | None."""
    monkeypatch.setattr(
        "app.workers.tasks.adjudicate_placement",
        lambda subject_title, candidates, settings=None: fn(subject_title, candidates),
    )


def _tree(driver, uid):
    with driver.session() as s:
        return {
            r["t"]: r["p"]
            for r in s.run(
                "MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task) "
                "OPTIONAL MATCH (t)-[:SUBGOAL_OF]->(p:Task) RETURN t.title AS t, p.title AS p",
                uid=uid,
            )
        }


def test_child_of_places_new_task_under_named_parent(client, driver, scoped_user, monkeypatch):
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Ship telemetry v2")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "Goal: ship telemetry v2.")

    _mock_adjudicator(monkeypatch, "Migrate telemetry schema")
    _mock_placement(
        monkeypatch,
        lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]) if c else None,
    )
    _ingest(client, f"{scoped_user}-s2", scoped_user, "Now migrating the telemetry schema.")

    assert _tree(driver, scoped_user) == {
        "Ship telemetry v2": None,
        "Migrate telemetry schema": "Ship telemetry v2",
    }


def test_parent_of_adopts_the_named_root(client, driver, scoped_user, monkeypatch):
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Fix duplicate column")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "Fixing the duplicate column.")

    _mock_adjudicator(monkeypatch, "Migrate telemetry schema")
    _mock_placement(
        monkeypatch,
        lambda t, c: PlacementVerdict(relation="parent_of", task_id=c[0][0]) if c else None,
    )
    _ingest(client, f"{scoped_user}-s2", scoped_user, "The bigger job is migrating the schema.")

    assert _tree(driver, scoped_user) == {
        "Fix duplicate column": "Migrate telemetry schema",
        "Migrate telemetry schema": None,
    }


def test_parent_of_a_non_root_writes_nothing(client, driver, scoped_user, monkeypatch):
    """Adopting an already-parented task would give it two parents."""
    from app.services.task_hierarchy import PlacementVerdict

    _mock_adjudicator(monkeypatch, "Root")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "root")
    _mock_adjudicator(monkeypatch, "Child")
    _mock_placement(monkeypatch, lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]))
    _ingest(client, f"{scoped_user}-s2", scoped_user, "child")

    # Third task tries to adopt "Child" (which has parent Root).
    _mock_adjudicator(monkeypatch, "Interloper")
    def adopt_child(t, c):
        cid = next(i for i, title in c if title == "Child")
        return PlacementVerdict(relation="parent_of", task_id=cid)
    _mock_placement(monkeypatch, adopt_child)
    _ingest(client, f"{scoped_user}-s3", scoped_user, "interloper")

    assert _tree(driver, scoped_user) == {"Root": None, "Child": "Root", "Interloper": None}


def test_none_and_malformed_verdicts_write_nothing(client, driver, scoped_user, monkeypatch):
    _mock_adjudicator(monkeypatch, "A")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "a")
    _mock_adjudicator(monkeypatch, "B")
    _mock_placement(monkeypatch, lambda t, c: None)  # model said none / parse failed
    _ingest(client, f"{scoped_user}-s2", scoped_user, "b")
    assert _tree(driver, scoped_user) == {"A": None, "B": None}


def test_already_parented_task_is_never_replaced(client, driver, scoped_user, monkeypatch):
    from app.services.task_hierarchy import PlacementVerdict
    from app.services.task_store import TaskStore

    _mock_adjudicator(monkeypatch, "Root")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "root")
    _mock_adjudicator(monkeypatch, "Other root")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s2", scoped_user, "other")
    _mock_adjudicator(monkeypatch, "Child")
    def under_root(t, c):
        rid = next(i for i, title in c if title == "Root")
        return PlacementVerdict(relation="child_of", task_id=rid)
    _mock_placement(monkeypatch, under_root)
    _ingest(client, f"{scoped_user}-s3", scoped_user, "child")

    # Continue "Child" (matched); placement is not even consulted.
    child_id = next(t.id for t in TaskStore(driver).list_open_tasks(scoped_user, limit=20) if t.title == "Child")
    _mock_adjudicator(monkeypatch, "Child", matches=child_id)
    called = {"n": 0}
    def spy(t, c):
        called["n"] += 1
        return PlacementVerdict(relation="child_of", task_id=next(i for i, title in c if title == "Other root"))
    _mock_placement(monkeypatch, spy)
    _ingest(client, f"{scoped_user}-s4", scoped_user, "more child work")

    assert called["n"] == 0, "placement must not run for an already-parented task"
    assert _tree(driver, scoped_user)["Child"] == "Root"


def test_hierarchy_error_does_not_fail_ingest(client, driver, scoped_user, monkeypatch):
    """set_parent raising must be swallowed at the worker boundary."""
    from app.services import task_store as ts

    _mock_adjudicator(monkeypatch, "A")
    _mock_placement(monkeypatch, lambda t, c: None)
    _ingest(client, f"{scoped_user}-s1", scoped_user, "a")

    def boom(self, child, parent):
        raise ts.TaskHierarchyError("simulated")
    monkeypatch.setattr(ts.TaskStore, "set_parent", boom)
    from app.services.task_hierarchy import PlacementVerdict
    _mock_adjudicator(monkeypatch, "B")
    _mock_placement(monkeypatch, lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]))
    _ingest(client, f"{scoped_user}-s2", scoped_user, "b")  # asserts 202 inside

    assert _tree(driver, scoped_user) == {"A": None, "B": None}


def test_tool_calls_are_stamped_with_the_leaf(client, driver, scoped_user, monkeypatch):
    from app.db.postgres import create_engine_from_settings
    from app.services.task_hierarchy import PlacementVerdict
    from app.services.workbench_store import recent_tool_calls, record_tool_call

    engine = create_engine_from_settings()
    try:
        _mock_adjudicator(monkeypatch, "Root")
        _mock_placement(monkeypatch, lambda t, c: None)
        _ingest(client, f"{scoped_user}-s1", scoped_user, "root")

        sid = f"{scoped_user}-s2"
        record_tool_call(engine, session_id=sid, user_id=scoped_user, tool_name="alembic", status="error", error="boom")
        _mock_adjudicator(monkeypatch, "Leaf")
        _mock_placement(monkeypatch, lambda t, c: PlacementVerdict(relation="child_of", task_id=c[0][0]))
        _ingest(client, sid, scoped_user, "leaf work")

        tree = _tree(driver, scoped_user)
        assert tree == {"Root": None, "Leaf": "Root"}
        from app.services.task_store import TaskStore
        leaf_id = next(t.id for t in TaskStore(driver).list_open_tasks(scoped_user, limit=20) if t.title == "Leaf")
        rows = recent_tool_calls(engine, session_id=sid, limit=10)
        assert rows and rows[0].task_id == leaf_id
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (scoped_user,))
        engine.dispose()
```

Check `recent_tool_calls`'s signature in `app/services/workbench_store.py:225` before running — it takes `engine, *, session_id=None, user_id=None, task_id=None, status=None, limit=...`; adjust the call if the kwarg names differ.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_task_worker.py -k "hierarchy or child_of or parent_of or leaf or replaced or verdicts" -v`
Expected: FAIL — `AttributeError: module 'app.workers.tasks' has no attribute 'adjudicate_placement'`.

- [ ] **Step 3: Implement**

In `app/workers/tasks.py` imports, add:

```python
from app.services.task_hierarchy import adjudicate_placement, shortlist_candidates
from app.services.task_store import TaskHierarchyError, TaskStore
```

(replace the existing `from app.services.task_store import TaskStore`).

Add after `_write_task`:

```python
def _title_similarity(embedder: Any):
    """Cosine over MiniLM embeddings of two short titles. The embedder is the
    worker's already-loaded model; the shortlist takes a callable so tests can
    inject a table."""
    import numpy as np

    def sim(a: str, b: str) -> float:
        va, vb = embedder.encode([a, b], normalize_embeddings=True)
        return float(np.dot(va, vb))

    return sim


def _write_hierarchy(
    neo_driver: Any,
    *,
    user_id: str,
    task_id: str,
    embedder: Any,
) -> None:
    """Place `task_id` in the user's goal tree, if the evidence and the model
    agree. Best-effort by contract — any failure logs and leaves the tree as
    it was. Runs only for a root subject; a parented Task is never re-placed.
    """
    try:
        store = TaskStore(neo_driver)
        if store.get_parent(task_id) is not None:
            return
        subject = store.task_evidence(task_id)
        if subject is None:
            return
        pool = store.list_placement_candidates(
            user_id, subject_id=task_id, limit=settings.task_candidate_limit
        )
        short = shortlist_candidates(
            subject,
            pool,
            similarity=_title_similarity(embedder),
            limit=settings.task_placement_candidates,
            min_score=settings.task_placement_min_score,
        )
        if not short:
            return
        verdict = adjudicate_placement(subject.title, [(c.id, c.title) for c in short])
        if verdict is None:
            return
        by_id = {c.id: c for c in short}
        if verdict.relation == "child_of":
            store.set_parent(task_id, verdict.task_id)
        else:  # parent_of: adopt exactly one root
            target = by_id.get(verdict.task_id)
            if target is None or not target.is_root:
                logger.info("placement wanted to adopt non-root %s; skipping", verdict.task_id)
                return
            store.set_parent(verdict.task_id, task_id)
        logger.info("hierarchy: %s %s %s", task_id, verdict.relation, verdict.task_id)
    except TaskHierarchyError as exc:
        logger.warning("placement rejected by hierarchy invariant: %s", exc)
    except Exception:
        logger.exception("hierarchy placement failed; continuing without it")
```

In `process_conversation`, at **both** call sites, immediately after `resolved_task_id = _write_task(...)` / `main_task_id = _write_task(...)` and still inside the `if user_id:` block:

```python
                if resolved_task_id:
                    _write_hierarchy(
                        neo_driver, user_id=user_id, task_id=resolved_task_id, embedder=embedder
                    )
```

and

```python
            if main_task_id:
                _write_hierarchy(
                    neo_driver, user_id=user_id, task_id=main_task_id, embedder=embedder
                )
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_task_worker.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/tasks.py tests/test_task_worker.py
git commit -m "feat: worker places each root task in the goal tree via shortlist + placement call

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `failed_calls` — lineage scope and rank

**Files:**
- Modify: `app/services/workbench_store.py:295-330`
- Test: `tests/test_workbench_store.py`

**Interfaces:**
- Produces: `failed_calls(engine, *, user_id, task_ids: list[str] | None = None, task_id: str | None = None, limit=5)`. `task_ids[0]` is the leaf (rank 2), the rest of the lineage rank 1, everything else 0; then recency. `task_id` is accepted and folded into a one-element list.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workbench_store.py`:

```python
def test_failed_calls_ranks_leaf_then_lineage_then_user(engine, ids):
    """Lineage scope: the leaf's failures lead, then any ancestor's, then the
    user's unrelated ones — union, never switch."""
    leaf, parent, other = ids["task"], ids["task_other"], "unrelated-task"
    e_user = record_tool_call(engine, session_id=ids["session"], user_id=ids["user"],
                              tool_name="u", status="error", error="u")
    e_other = record_tool_call(engine, session_id=ids["session"], user_id=ids["user"],
                               tool_name="o", status="error", error="o", task_id=other)
    e_parent = record_tool_call(engine, session_id=ids["session"], user_id=ids["user"],
                                tool_name="p", status="error", error="p", task_id=parent)
    e_leaf = record_tool_call(engine, session_id=ids["session"], user_id=ids["user"],
                              tool_name="l", status="error", error="l", task_id=leaf)

    rows = failed_calls(engine, user_id=ids["user"], task_ids=[leaf, parent])
    assert [r.id for r in rows] == [e_leaf.id, e_parent.id, e_other.id, e_user.id]

    # Back-compat: a single task_id behaves like a one-element lineage.
    rows1 = failed_calls(engine, user_id=ids["user"], task_id=parent)
    assert rows1[0].id == e_parent.id
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workbench_store.py -k lineage -v`
Expected: FAIL — `TypeError: failed_calls() got an unexpected keyword argument 'task_ids'`.

- [ ] **Step 3: Implement**

Replace `failed_calls`:

```python
def failed_calls(
    engine: Engine,
    *,
    user_id: str,
    task_ids: list[str] | None = None,
    task_id: str | None = None,
    limit: int = 5,
) -> list[ToolCallRow]:
    """Most recent failed calls, lineage-ranked: the leaf task's own failures
    first, then any ancestor's, then the user's other failures by recency.

    `task_ids` is the active lineage `[leaf, parent, grandparent, ...]`;
    `task_id` (legacy) is folded into a one-element lineage. Scope UNIONS —
    it never switches. Switching was a live bug: a failure stamped to an
    older task vanished the moment any unrelated newer task became active.
    The rank CASE is COALESCE-free because `task_id IN (...)` on a NULL
    yields NULL, and `CASE WHEN NULL` falls through to ELSE 0 — untasked
    failures rank last, which is what we want.
    """
    if limit <= 0:
        return []
    lineage = list(task_ids or ([task_id] if task_id else []))
    leaf = lineage[0] if lineage else None
    # An empty IN-list is invalid SQL; use a value no task id can equal.
    in_list = lineage or ["__none__"]
    sql = text(
        f"""
        SELECT {_COLUMNS} FROM tool_calls
        WHERE status = 'error'
          AND (user_id = :user_id OR task_id IN :lineage)
        ORDER BY CASE WHEN task_id = :leaf THEN 2
                      WHEN task_id IN :lineage THEN 1
                      ELSE 0 END DESC,
                 created_at DESC, id DESC
        LIMIT :limit
        """
    ).bindparams(bindparam("lineage", expanding=True))
    params = {"user_id": user_id, "lineage": in_list, "leaf": leaf, "limit": limit}
    with engine.connect() as conn:
        return [_to_row(r) for r in conn.execute(sql, params)]
```

Add `bindparam` to the sqlalchemy import: `from sqlalchemy import bindparam, text`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_workbench_store.py -v`
Expected: all PASS (including the existing union test — its ordering `[e1, e3, e2]` still holds: e1 rank 2, e3 and e2 rank 0 by recency).

- [ ] **Step 5: Commit**

```bash
git add app/services/workbench_store.py tests/test_workbench_store.py
git commit -m "feat: failed_calls scopes and ranks by task lineage

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Lineage Task seeds — `build_seeds(task_nodes=)` and `fetch_neighborhood(task_ids=)`

**Files:**
- Modify: `app/services/proactive.py` (`build_seeds`)
- Modify: `app/services/neo4j_store.py` (`fetch_neighborhood`)
- Test: `tests/test_proactive_assembly.py`, `tests/test_neighborhood_task_seeds.py` (new, integration)

**Interfaces:**
- Produces: `build_seeds(..., task_nodes: dict[str, float] | None = None)` — keys are `Task:<id>` node ids, merged max-wins; `Neo4jStore.fetch_neighborhood(entity_names, *, task_ids: list[str] | None = None, radius=4)`; `lineage_task_seeds(lineage_ids: list[str], *, base: float, decay: float) -> dict[str, float]` in `proactive.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_proactive_assembly.py`:

```python
def test_task_nodes_seed_directly_and_merge_max_wins():
    seeds = build_seeds(
        live_entities=["clickhouse"],
        task_entities=[],
        alias_to_profile={},
        task_nodes={"Task:leaf": 0.2, "Task:parent": 0.14},
    )
    assert seeds["Task:leaf"] == pytest.approx(0.2)
    assert seeds["Task:parent"] == pytest.approx(0.14)
    assert seeds["Entity:clickhouse"] == 1.0


def test_lineage_task_seeds_decay_by_depth():
    from app.services.proactive import lineage_task_seeds

    seeds = lineage_task_seeds(["leaf", "mid", "root", "great"], base=0.2, decay=0.7)
    assert seeds == {
        "Task:leaf": pytest.approx(0.2),
        "Task:mid": pytest.approx(0.14),
        "Task:root": pytest.approx(0.098),
        "Task:great": pytest.approx(0.0686),
    }
    assert lineage_task_seeds([], base=0.2, decay=0.7) == {}


def test_spec_4c_arithmetic_is_pinned():
    """The design's claim: from a lineage seed, Task->ADVANCES->Episode->INVOKED
    ->ToolCall lands leaf 0.104 / parent 0.073 / grandparent 0.051 (> 0.05
    floor), depth-3 episode 0.049 dies; and a live entity's episode (0.164)
    still outranks the leaf's own (0.144)."""
    from app.services.activation import spread_activation
    from app.services.proactive import lineage_task_seeds

    nb = _nb(
        ("Task:leaf", "SUBGOAL_OF", "Task:mid", 1),
        ("Task:mid", "SUBGOAL_OF", "Task:root", 1),
        ("Task:root", "SUBGOAL_OF", "Task:great", 1),
        ("Episode:1", "ADVANCES", "Task:leaf", 1),
        ("Episode:2", "ADVANCES", "Task:mid", 1),
        ("Episode:3", "ADVANCES", "Task:root", 1),
        ("Episode:4", "ADVANCES", "Task:great", 1),
        ("Episode:1", "INVOKED", "ToolCall:1", 1),
        ("Episode:2", "INVOKED", "ToolCall:2", 1),
        ("Episode:3", "INVOKED", "ToolCall:3", 1),
        ("Episode:9", "MENTIONS", "Entity:live", 1),
    )
    seeds = {**lineage_task_seeds(["leaf", "mid", "root", "great"], base=0.2, decay=0.7),
             "Entity:live": 1.0}
    res = spread_activation(nb, seeds=seeds, floor=0.05, decay=0.8)
    s = res.scores
    assert s["Episode:1"] == pytest.approx(0.144, abs=1e-3)
    assert s["ToolCall:1"] == pytest.approx(0.104, abs=1e-3)
    assert s["ToolCall:2"] == pytest.approx(0.073, abs=1e-3)
    assert s["ToolCall:3"] == pytest.approx(0.051, abs=1e-3)
    assert "Episode:4" not in s, "depth-3 episode must fall under the floor"
    assert s["Episode:9"] == pytest.approx(0.164, abs=1e-3)
    assert s["Episode:9"] > s["Episode:1"], "live evidence must outrank the goal's own history"
```

Create `tests/test_neighborhood_task_seeds.py`:

```python
"""fetch_neighborhood can start from Task ids as well as Entity names."""

from __future__ import annotations

import uuid

import pytest

from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.services.neo4j_store import Neo4jStore
from app.services.task_store import TaskStore

pytestmark = pytest.mark.integration


@pytest.fixture
def driver():
    d = create_driver_from_settings()
    ensure_constraints(d)
    yield d
    d.close()


def test_neighborhood_from_task_ids_reaches_parent_episode_and_tool_call(driver):
    uid = f"nb-{uuid.uuid4().hex[:8]}"
    ts = TaskStore(driver)
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $u})", u=uid)
    root = ts.create_task(uid, "root")
    leaf = ts.create_task(uid, "leaf")
    ts.set_parent(leaf, root)
    try:
        with driver.session() as s:
            s.run(
                "MATCH (t:Task {id: $root}) MERGE (e:Episode {id: -940001}) MERGE (e)-[:ADVANCES]->(t) "
                "MERGE (tc:ToolCall {id: -940002}) SET tc.tool_name='alembic', tc.status='error' "
                "MERGE (e)-[:INVOKED]->(tc)",
                root=root,
            )
        nb = Neo4jStore(driver).fetch_neighborhood([], task_ids=[leaf, root], radius=4)
        ids = {e.src for e in nb.edges} | {e.dst for e in nb.edges}
        assert f"Task:{leaf}" in ids and f"Task:{root}" in ids
        assert "Episode:-940001" in ids and "ToolCall:-940002" in ids
        assert nb.labels[f"Task:{root}"] == "Task"
        rels = {e.rel for e in nb.edges}
        assert {"SUBGOAL_OF", "ADVANCES", "INVOKED"} <= rels
        # Still empty when given nothing at all.
        assert Neo4jStore(driver).fetch_neighborhood([], task_ids=[]).edges == []
    finally:
        with driver.session() as s:
            s.run("MATCH (n) WHERE n:Episode AND n.id = -940001 DETACH DELETE n")
            s.run("MATCH (n) WHERE n:ToolCall AND n.id = -940002 DETACH DELETE n")
            s.run("MATCH (p:UserProfile {user_id: $u}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) DETACH DELETE p, t", u=uid)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_proactive_assembly.py tests/test_neighborhood_task_seeds.py -k "task_nodes or lineage or 4c or task_ids" -v`
Expected: FAIL (`task_nodes` unexpected kwarg; `lineage_task_seeds` missing; `task_ids` unexpected kwarg).

- [ ] **Step 3: Implement**

In `app/services/proactive.py`:

```python
def lineage_task_seeds(
    lineage_ids: list[str], *, base: float, decay: float
) -> dict[str, float]:
    """`Task:<id>` seeds for `[leaf, parent, grandparent, ...]` at
    `base * decay ** depth`. Structural pull up the tree — see spec §4c for
    why the Task node (all-structural edges to its episodes and tool calls)
    and not the ancestors' entities (a counted MENTIONS hop lands at the
    floor)."""
    return {f"Task:{tid}": base * (decay ** d) for d, tid in enumerate(lineage_ids)}
```

`build_seeds` — add the parameter and merge at the end:

```python
def build_seeds(
    *,
    live_entities: list[str],
    task_entities: list[str],
    alias_to_profile: dict[str, str],
    task_seed: float = DEFAULT_TASK_SEED,
    task_nodes: dict[str, float] | None = None,
) -> dict[str, float]:
    ...  # existing body unchanged, then before `return seeds`:
    for nid, a in (task_nodes or {}).items():
        seeds[nid] = max(seeds.get(nid, 0.0), a)
    return seeds
```

Update the docstring with one bullet: `- Task nodes (the active lineage) seed at the given activations; see lineage_task_seeds.`

In `app/services/neo4j_store.py` `fetch_neighborhood`:

```python
    def fetch_neighborhood(
        self,
        entity_names: list[str],
        *,
        task_ids: list[str] | None = None,
        radius: int = 4,
    ):
        """Pull the subgraph within `radius` hops of the named entities and/or
        the given Task ids. ... (keep existing docstring text) ...
        """
        from app.services.activation import Edge, Neighborhood

        names = [normalize_entity_name(n) for n in entity_names]
        names = [n for n in names if n]
        tids = [t for t in (task_ids or []) if t]
        if not names and not tids:
            return Neighborhood()
        r = max(1, min(int(radius), 6))
        q = f"""
        MATCH (seed)
        WHERE (seed:Entity AND seed.name IN $names)
           OR (seed:Task AND seed.id IN $task_ids)
        MATCH p = (seed)-[*1..{r}]-(other)
        UNWIND relationships(p) AS rel
        WITH DISTINCT rel, startNode(rel) AS a, endNode(rel) AS b
        RETURN labels(a)[0] AS la,
               coalesce(a.name, a.user_id, toString(a.id)) AS ka,
               type(rel) AS rel_type,
               coalesce(rel.count, 1) AS cnt,
               labels(b)[0] AS lb,
               coalesce(b.name, b.user_id, toString(b.id)) AS kb
        """
        edges: list = []
        labels: dict[str, str] = {}
        with self._driver.session() as session:
            for rec in session.run(q, names=names, task_ids=tids):
                ...  # unchanged
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_proactive_assembly.py tests/test_neighborhood_task_seeds.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/proactive.py app/services/neo4j_store.py tests/test_proactive_assembly.py tests/test_neighborhood_task_seeds.py
git commit -m "feat: lineage Task nodes seed activation; neighborhood fetch starts from Task ids

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Retrieval — lineage in all three call sites

**Files:**
- Modify: `app/services/retrieval.py` (`_format_profile_facts`, `_format_known_failures`, `_format_proactive_context`)
- Create: `tests/test_goal_hierarchy_retrieval.py`

**Interfaces:**
- Consumes: `TaskStore.active_task/get_ancestors/get_lineage_ids`, `failed_calls(task_ids=)`, `lineage_task_seeds`, `build_seeds(task_nodes=)`, `fetch_neighborhood(task_ids=)`.
- Produces: `Current task:` line `"{leaf} (under: {parent} ▸ {grandparent})"`, source `task` with `lineage: list[str]`, `depth: int`; Known Failures lineage-ranked; proactive sources whose `path` starts at a `Task:` seed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_goal_hierarchy_retrieval.py`:

```python
"""Retrieval pulls context up the goal hierarchy. Hand-built graph, live stack.

Three effects, each asserted independently against a three-level tree
(leaf <- mid <- root) where only the ROOT has a failing tool call:
  1. `Current task:` shows the path.
  2. Known Failures surfaces the root's failure ranked by lineage.
  3. Proactive context surfaces the root's ToolCall via the lineage Task seed,
     with a provenance path that starts at the root Task node.
Plus two severed tests proving each mechanism is the one doing the work.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.config import settings
from app.db.neo4j import create_driver_from_settings, ensure_constraints
from app.db.postgres import create_engine_from_settings, ensure_l2_schema, session_scope
from app.services.neo4j_store import Neo4jStore
from app.services.postgres_store import PostgresStore
from app.services.retrieval import retrieve_context
from app.services.task_store import TaskStore
from app.services.workbench_store import claim_tool_calls, ensure_l4_schema, record_tool_call
from tests.conftest import unit_embedding_384

pytestmark = pytest.mark.integration


@pytest.fixture
def stack():
    engine = create_engine_from_settings()
    ensure_l2_schema(engine)
    ensure_l4_schema(engine)
    driver = create_driver_from_settings()
    ensure_constraints(driver)
    yield engine, driver
    driver.close()
    engine.dispose()


@pytest.fixture
def tree(stack, monkeypatch):
    """root <- mid <- leaf. Root's episode invoked a failing `alembic` call.
    Live session has one L1 message with no NER entities at all, so the ONLY
    routes to the failure are the lineage (Known Failures) and the Task seeds
    (proactive)."""
    engine, driver = stack
    uid = f"gh-{uuid.uuid4().hex[:8]}"
    root_s, mid_s, leaf_s, live_s = (f"{uid}-{x}" for x in ("root", "mid", "leaf", "live"))
    now = datetime.now(timezone.utc)
    ts = TaskStore(driver)
    with driver.session() as s:
        s.run("MERGE (:UserProfile {user_id: $u})", u=uid)
    root = ts.create_task(uid, "Ship telemetry v2")
    mid = ts.create_task(uid, "Migrate telemetry schema")
    leaf = ts.create_task(uid, "Fix duplicate column on episodes")
    ts.set_parent(mid, root)
    ts.set_parent(leaf, mid)

    def episode(sid, summary, axis):
        with session_scope(engine) as s:
            return PostgresStore(s).insert_episode(
                session_id=sid, summary=summary, embedding=unit_embedding_384(primary_axis=axis),
                start_time=now, end_time=now, metadata=None, user_id=uid,
            )
    e_root = episode(root_s, "Kicked off shipping telemetry v2.", 301)
    e_mid = episode(mid_s, "Started migrating the telemetry schema.", 302)
    e_leaf = episode(leaf_s, "Looking at the duplicate column.", 303)
    call = record_tool_call(engine, session_id=root_s, user_id=uid, task_id=root,
                            tool_name="alembic", status="error",
                            error="DuplicateColumn: user_id already exists")
    claim_tool_calls(engine, session_id=root_s, episode_id=e_root, task_id=root)

    g = Neo4jStore(driver)
    for sid, eid, summ in ((root_s, e_root, "root"), (mid_s, e_mid, "mid"), (leaf_s, e_leaf, "leaf")):
        g.upsert_session(sid)
        g.upsert_episode(sid, eid, summ)
    ts.link_episode(root, episode_id=e_root)
    ts.link_episode(mid, episode_id=e_mid)
    ts.link_episode(leaf, episode_id=e_leaf)   # leaf is now active
    g.link_tool_calls(e_root, [(call.id, "alembic", "error", now.isoformat())])

    from app.api import services as api_services
    api_services.get_redis_store().append_messages(
        live_s, [{"role": "user", "content": "ok, picking this back up."}]
    )
    monkeypatch.setattr(
        "app.api.services.get_query_embedder",
        lambda: type("E", (), {"encode": lambda self, t, normalize_embeddings=True: type(
            "V", (), {"tolist": lambda s: unit_embedding_384(primary_axis=7)})()})(),
    )
    yield {"uid": uid, "live": live_s, "root": root, "mid": mid, "leaf": leaf, "call_id": call.id}

    api_services.get_redis_client().delete(f"session:{live_s}")
    with engine.begin() as c:
        c.exec_driver_sql("DELETE FROM tool_calls WHERE user_id = %s", (uid,))
        c.exec_driver_sql("DELETE FROM episodes WHERE user_id = %s", (uid,))
    with driver.session() as s:
        s.run("MATCH (p:UserProfile {user_id:$u}) OPTIONAL MATCH (p)-[:PURSUES]->(t:Task) DETACH DELETE p, t", u=uid)
        for sid in (root_s, mid_s, leaf_s, live_s):
            s.run("MATCH (se:Session {id:$s}) OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep) "
                  "OPTIONAL MATCH (ep)-[:INVOKED]->(tc) DETACH DELETE se, ep, tc", s=sid)


def _retrieve(tree):
    return retrieve_context(tree["live"], "what should I keep in mind?", 2000, tree["uid"])


def test_current_task_line_shows_the_path(tree):
    r = _retrieve(tree)
    ctx = r["context"]
    assert "Current task: Fix duplicate column on episodes (under: Migrate telemetry schema ▸ Ship telemetry v2)" in ctx
    (src,) = [s for s in r["sources"] if s["type"] == "task"]
    assert src["details"]["lineage"] == [tree["leaf"], tree["mid"], tree["root"]]
    assert src["details"]["depth"] == 2


def test_closed_ancestor_renders_done(tree, stack):
    _, driver = stack
    TaskStore(driver).close_task(tree["root"])
    ctx = _retrieve(tree)["context"]
    assert "Ship telemetry v2 (done)" in ctx


def test_known_failures_surface_the_roots_failure(tree):
    r = _retrieve(tree)
    fails = [s for s in r["sources"] if s["type"] == "tool_failure"]
    assert fails and fails[0]["details"]["task_id"] == tree["root"]
    assert "DuplicateColumn" in r["context"]


def test_proactive_surfaces_root_tool_call_via_lineage_seed(tree):
    r = _retrieve(tree)
    pf = [s for s in r["sources"] if s["type"] == "proactive_tool_failure"]
    assert pf, f"root's failure did not surface proactively; types={sorted({s['type'] for s in r['sources']})}"
    d = pf[0]["details"]
    assert d["tool_call_id"] == tree["call_id"]
    assert d["path"], "no provenance path"
    first_src = d["path"][0][0]
    assert first_src == f"Task:{tree['root']}", f"path should start at the root Task seed, got {first_src}"
    # depth-2 tool call lands just above the floor per spec §4c
    assert 0.05 <= d["activation"] < 0.06


def test_severed_lineage_seeds_lose_the_proactive_failure(tree, monkeypatch):
    monkeypatch.setattr(settings, "proactive_task_node_seed", 0.0)
    r = _retrieve(tree)
    assert not [s for s in r["sources"] if s["type"] == "proactive_tool_failure"], (
        "with Task seeds off the failure must not surface proactively — something else carried it"
    )
    # ...but Known Failures (deterministic lineage) still has it.
    assert [s for s in r["sources"] if s["type"] == "tool_failure"]


def test_severed_subgoal_prior_still_reaches_via_direct_seed(tree, monkeypatch):
    """Direct Task seeds don't need SUBGOAL_OF to reach their own episodes; this
    pins that the *seed* is the mechanism, and the prior is what carries paths
    that must cross the tree (exercised by the agentic sibling scenario)."""
    from app.services import activation
    monkeypatch.setitem(activation.EDGE_PRIORS, "SUBGOAL_OF", 0.0)
    r = _retrieve(tree)
    assert [s for s in r["sources"] if s["type"] == "proactive_tool_failure"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goal_hierarchy_retrieval.py -v`
Expected: FAIL — `Current task:` line has no `(under: …)`; no `proactive_tool_failure` source; `task` source lacks `lineage`.

- [ ] **Step 3: Implement**

In `retrieval.py` `_format_profile_facts`, replace the "Most recently active open task" block:

```python
    # The active task — the leaf being worked — rendered with its ancestor path
    # so the agent knows the larger objective it is serving. One line, one
    # source (the section merger pairs lines and sources 1:1).
    from app.services.task_store import TaskStore

    task_store = TaskStore(api_services.get_neo4j_driver())
    active = task_store.active_task(user_id)
    if active is not None:
        ancestors = task_store.get_ancestors(active.id)
        line = f"Current task: {active.title}"
        if ancestors:
            chain = " ▸ ".join(
                a.title + (" (done)" if a.status == "done" else "") for a in ancestors
            )
            line += f" (under: {chain})"
        lines.append(line)
        sources.append(
            _source(
                "task",
                task_id=active.id,
                title=active.title,
                status=active.status,
                lineage=[active.id] + [a.id for a in ancestors],
                depth=len(ancestors),
            )
        )
```

`_format_known_failures` — replace the active-task lookup and the call:

```python
    task_store = TaskStore(api_services.get_neo4j_driver())
    active = task_store.active_task(user_id)
    lineage = task_store.get_lineage_ids(active.id) if active else []

    rows = failed_calls(
        engine,
        user_id=user_id,
        task_ids=lineage or None,
        limit=settings.workbench_max_failures_in_context,
    )
```

`_format_proactive_context` — replace step 2 and the neighborhood fetch:

```python
    # 2. Inherited seeds: what the active task already touches (topical, leaf
    #    only), plus the lineage Task nodes themselves (structural, decaying up
    #    the tree — see spec §4c).
    from app.services.proactive import lineage_task_seeds

    task_store = TaskStore(driver)
    task_entities: list[str] = []
    lineage: list[str] = []
    active_task = task_store.active_task(user_id)
    if active_task is not None:
        lineage = task_store.get_lineage_ids(active_task.id)
        with driver.session() as sess:
            task_entities = [
                r["name"]
                for r in sess.run(
                    "MATCH (:Task {id: $tid})<-[:ADVANCES]-(:Episode)-[:MENTIONS]->(e:Entity) "
                    "RETURN DISTINCT e.name AS name",
                    tid=active_task.id,
                )
            ]
    task_nodes = lineage_task_seeds(
        lineage,
        base=settings.proactive_task_node_seed,
        decay=settings.proactive_task_depth_decay,
    )

    # 3. Alias collapse ...
    seeds = build_seeds(
        live_entities=live,
        task_entities=task_entities,
        alias_to_profile=alias_to_profile,
        task_seed=settings.proactive_task_seed,
        task_nodes=task_nodes,
    )
    if not seeds:
        return [], []

    # 4. One round-trip neighborhood (entities + lineage tasks), spread in memory.
    entity_seed_names = [nid.split(":", 1)[1] for nid in seeds if nid.startswith("Entity:")]
    task_seed_ids = [nid.split(":", 1)[1] for nid in seeds if nid.startswith("Task:")]
    neighborhood = graph.fetch_neighborhood(
        entity_seed_names, task_ids=task_seed_ids, radius=settings.proactive_fetch_radius
    )
```

Also drop the now-unused `TaskStore` re-import inside the `Task` render branch (use the `task_store` local).

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goal_hierarchy_retrieval.py tests/test_proactive_retrieval.py tests/test_retrieval.py tests/test_retrieval_profile.py -v`
Expected: all PASS. If `test_proactive_surfaces_root_tool_call_via_lineage_seed`'s activation bound fails, print `d["activation"]` and re-derive: root at depth 2 → `0.2·0.49 = 0.098`; episode `0.098·0.9·0.8 = 0.0706`; tool call `0.0706·0.72 = 0.0508`. Adjust the assertion window only if the arithmetic (not the code) was wrong.

- [ ] **Step 5: Commit**

```bash
git add app/services/retrieval.py tests/test_goal_hierarchy_retrieval.py
git commit -m "feat: retrieval pulls context up the goal lineage — path line, failures, Task seeds

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Agentic harness — four three-session scenarios against real Ollama

**Files:**
- Create: `tests/agentic/hierarchy_scenarios.py`
- Create: `tests/agentic/test_goal_hierarchy.py`

**Interfaces:**
- Consumes: `OllamaAgent`, `Turn`, fixtures `agent`, `api`, `auth`, `ingest`, `retrieve`, `profile_user_id`, `neo4j_driver`, `pg_engine`.
- Produces: `HierarchyScenario`, `SessionPlan`, `HIERARCHY_SCENARIOS`; a `hierarchy_run` helper that plays a scenario and returns `(context, sources, tree)`.

- [ ] **Step 1: Write the scenarios**

Create `tests/agentic/hierarchy_scenarios.py`:

```python
"""Three-session goal-hierarchy scenarios with declared ground truth.

Same contract as `scenarios.py`: the model chooses the words, the scenario
chooses the facts. Each session is a plan (a Turn plus an optional failing tool
call recorded before ingest). Session 3 ends with a retrieval whose query
deliberately names none of the earlier facts — anything the agent learns, it
learned from the hierarchy.

`gate_*` fields are asserted. `metric_*` fields are printed under [metric].
Which is which was decided by the gate-promotion rule in the spec: a
behavioural claim gates only if it held on every trial during implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.agentic.ollama_agent import Turn


@dataclass(frozen=True)
class SessionPlan:
    turn: Turn
    tool_failure: dict | None = None   # {"tool_name","error","args"} recorded before ingest


@dataclass(frozen=True)
class HierarchyScenario:
    name: str
    sessions: list[SessionPlan]        # exactly three
    final_query: str
    #: substrings that MUST appear in S3's context (lowercased compare)
    gate_contains: list[str] = field(default_factory=list)
    #: substrings that MUST NOT appear in S3's context
    gate_absent: list[str] = field(default_factory=list)
    #: (child_title_fragment, parent_title_fragment) pairs — reported as [metric]
    metric_tree: list[tuple[str, str]] = field(default_factory=list)
    #: title fragments that must NOT have a parent — reported as [metric]
    metric_roots: list[str] = field(default_factory=list)
    #: substrings reported (not asserted) in S3's context
    metric_contains: list[str] = field(default_factory=list)


ALEMBIC_FAILURE = {
    "tool_name": "alembic",
    "args": {"command": "upgrade", "revision": "0042"},
    "error": "DuplicateColumn: column user_id already exists on episodes",
}


TOP_DOWN = HierarchyScenario(
    name="top_down",
    sessions=[
        SessionPlan(Turn(
            intent="State plainly that your overall goal this quarter is to ship telemetry v2.",
            anchors=["ship telemetry v2"],
            fallback="My overall goal this quarter is to ship telemetry v2.",
        )),
        SessionPlan(Turn(
            intent="Say that as part of shipping telemetry v2 you are now migrating the telemetry schema to ClickHouse, and the alembic migration just failed.",
            anchors=["migrate the telemetry schema", "ClickHouse", "alembic"],
            fallback="As part of shipping telemetry v2 I need to migrate the telemetry schema to ClickHouse, and the alembic migration just failed.",
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent="Say your goal right now is to fix the duplicate user_id column on the episodes table, a step of the schema migration.",
            anchors=["fix the duplicate", "episodes"],
            fallback="Right now my goal is to fix the duplicate user_id column on the episodes table, which is a step of the schema migration.",
        )),
    ],
    final_query="Before I start, is there anything from earlier I should keep in mind?",
    gate_contains=["duplicatecolumn"],
    metric_tree=[("duplicate", "migrat"), ("migrat", "telemetry v2")],
    metric_contains=["telemetry v2", "under:"],
)


BOTTOM_UP = HierarchyScenario(
    name="bottom_up",
    sessions=[
        SessionPlan(Turn(
            intent="Say your goal is to fix the duplicate user_id column on the episodes table, and that the alembic upgrade just failed.",
            anchors=["fix the duplicate", "episodes", "alembic"],
            fallback="My goal is to fix the duplicate user_id column on the episodes table; the alembic upgrade just failed.",
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent="Say that the bigger goal, which the duplicate-column fix is one step of, is migrating the whole telemetry schema to ClickHouse.",
            anchors=["migrate", "telemetry schema", "ClickHouse"],
            fallback="The bigger goal here — the duplicate column fix is just one step of it — is to migrate the whole telemetry schema to ClickHouse.",
        )),
        SessionPlan(Turn(
            intent="Say you are continuing the telemetry schema migration today.",
            anchors=["telemetry schema migration"],
            fallback="Continuing the telemetry schema migration today.",
        )),
    ],
    final_query="What should I watch out for?",
    gate_contains=["duplicatecolumn"],
    metric_tree=[("duplicate", "migrat")],
    metric_contains=["under:"],
)


UNRELATED_STAYS_FLAT = HierarchyScenario(
    name="unrelated_stays_flat",
    sessions=[
        SessionPlan(Turn(
            intent="State that your goal is to ship telemetry v2 and that an alembic migration failed.",
            anchors=["ship telemetry v2", "alembic"],
            fallback="My goal is to ship telemetry v2; an alembic migration failed today.",
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent="State that your goal is to plan the team offsite in Lisbon for October.",
            anchors=["plan the team offsite", "Lisbon"],
            fallback="My goal is to plan the team offsite in Lisbon for October.",
        )),
        SessionPlan(Turn(
            intent="Say you are continuing to plan the Lisbon offsite and looking at venues.",
            anchors=["Lisbon", "offsite"],
            fallback="Continuing to plan the Lisbon offsite; looking at venues.",
        )),
    ],
    final_query="Anything I should keep in mind?",
    # Precision: the telemetry goal must NOT be reported as an ancestor.
    gate_absent=["under: ship telemetry v2", "under: ship telemetry"],
    metric_roots=["telemetry", "offsite"],
)


SIBLING_SUBGOALS = HierarchyScenario(
    name="sibling_subgoals",
    sessions=[
        SessionPlan(Turn(
            intent="State plainly that your overall goal is to ship telemetry v2.",
            anchors=["ship telemetry v2"],
            fallback="My overall goal is to ship telemetry v2.",
        )),
        SessionPlan(Turn(
            intent="Say that one part of shipping telemetry v2 is migrating the telemetry schema with alembic, and it just failed.",
            anchors=["telemetry v2", "migrat", "alembic"],
            fallback="One part of shipping telemetry v2 is migrating the telemetry schema with alembic, and it just failed.",
        ), tool_failure=ALEMBIC_FAILURE),
        SessionPlan(Turn(
            intent="Say that another part of shipping telemetry v2 is building the Grafana dashboards, and you are starting that now.",
            anchors=["telemetry v2", "Grafana dashboards"],
            fallback="Another part of shipping telemetry v2 is building the Grafana dashboards; starting that now.",
        )),
    ],
    final_query="Anything I should know before I dig in?",
    gate_contains=[],
    metric_tree=[("migrat", "telemetry v2"), ("grafana", "telemetry v2")],
    metric_contains=["under:", "duplicatecolumn"],
)


HIERARCHY_SCENARIOS = [TOP_DOWN, BOTTOM_UP, UNRELATED_STAYS_FLAT, SIBLING_SUBGOALS]
```

- [ ] **Step 2: Write the test**

Create `tests/agentic/test_goal_hierarchy.py`:

```python
"""Goal hierarchy under real traffic: three sessions, real Ollama for both the
agent's words AND MemCache's adjudication + placement, real stack.

Every test prints S3's retrieved context under [context], the user's task tree
under [tree], and the model's judgement calls under [metric], so the run can be
read qualitatively as well as pass/fail. Gates are the claims the deterministic
suite already proves the plumbing for; the tree shape is a metric.
"""

from __future__ import annotations

import pytest

from tests.agentic.hierarchy_scenarios import HIERARCHY_SCENARIOS, HierarchyScenario
from tests.agentic.ollama_agent import summarize_anchor_recall

pytestmark = [pytest.mark.agentic, pytest.mark.integration]


def _tree(driver, uid) -> list[tuple[str, str, str | None]]:
    """[(title, status, parent_title)] for every task the user pursues."""
    with driver.session() as s:
        return [
            (r["t"], r["st"], r["p"])
            for r in s.run(
                "MATCH (:UserProfile {user_id: $uid})-[:PURSUES]->(t:Task) "
                "OPTIONAL MATCH (t)-[:SUBGOAL_OF]->(p:Task) "
                "RETURN t.title AS t, t.status AS st, p.title AS p ORDER BY t.created_at",
                uid=uid,
            )
        ]


def _render_tree(rows) -> str:
    return "\n".join(f"  {t!r} ({st})  <- parent: {p!r}" for t, st, p in rows) or "  (no tasks)"


@pytest.fixture
def hierarchy_run(agent, api, auth, ingest, retrieve, profile_user_id, neo4j_driver, pg_engine, release_person_names):
    """Play a scenario; return (context, sources, tree_rows). Cleans up all
    three sessions afterwards (the profile fixture removes the tasks)."""
    made: list[str] = []

    def _run(sc: HierarchyScenario):
        release_person_names("dana whitfield", "dana")
        agent.transcript.clear()
        uid = profile_user_id
        base = f"agentic-h-{uid[-6:]}"
        for i, plan in enumerate(sc.sessions, start=1):
            sid = f"{base}-s{i}"
            made.append(sid)
            if plan.tool_failure:
                r = api.post("/workbench/tool-call", headers=auth, json={
                    "session_id": sid, "user_id": uid, "status": "error", **plan.tool_failure,
                })
                assert r.status_code == 201, r.text
            ingest(sid, agent.exchange(plan.turn), {"scenario": sc.name}, uid)
        result = retrieve(made[-1], sc.final_query, 2000, uid)
        rows = _tree(neo4j_driver, uid)
        recall = summarize_anchor_recall([p.turn for p in sc.sessions], agent.transcript)
        print(f"\n[{sc.name}] anchor_recall={recall:.2f}")
        print(f"[tree]\n{_render_tree(rows)}")
        print(f"[context]\n{result['context']}\n")
        return result["context"], result["sources"], rows

    yield _run

    import redis as _redis
    from app.config import settings as _settings
    c = _redis.from_url(_settings.redis_url, decode_responses=True)
    try:
        for sid in made:
            c.delete(f"session:{sid}")
    finally:
        c.close()
    with pg_engine.begin() as conn:
        for sid in made:
            conn.exec_driver_sql("DELETE FROM tool_calls WHERE session_id = %s", (sid,))
            conn.exec_driver_sql("DELETE FROM episodes WHERE session_id = %s", (sid,))
    with neo4j_driver.session() as s:
        for sid in made:
            s.run(
                "MATCH (se:Session {id: $sid}) OPTIONAL MATCH (se)-[:HAS_EPISODE]->(ep:Episode) "
                "OPTIONAL MATCH (ep)-[:DECIDED|PREFERS]->(dp) OPTIONAL MATCH (ep)-[:INVOKED]->(tc:ToolCall) "
                "DETACH DELETE se, ep, dp, tc",
                sid=sid,
            )


def _has_edge(rows, child_frag: str, parent_frag: str) -> bool:
    return any(
        child_frag in (t or "").lower() and parent_frag in (p or "").lower()
        for t, _, p in rows
    )


@pytest.mark.parametrize("scenario", HIERARCHY_SCENARIOS, ids=lambda s: s.name)
def test_hierarchy_scenario(scenario, hierarchy_run):
    context, sources, rows = hierarchy_run(scenario)
    low = context.lower()

    # ---- gates: behaviour the agent can rely on
    for frag in scenario.gate_contains:
        assert frag in low, f"[{scenario.name}] context missing {frag!r}"
    for frag in scenario.gate_absent:
        assert frag not in low, f"[{scenario.name}] context must not contain {frag!r} (false parent)"

    # ---- metrics: LLM judgement, reported never gated
    for child, parent in scenario.metric_tree:
        ok = _has_edge(rows, child, parent)
        print(f"[metric] {scenario.name}: edge {child!r} -SUBGOAL_OF-> {parent!r}: {'yes' if ok else 'NO'}")
    for frag in scenario.metric_roots:
        is_root = any(frag in (t or "").lower() and p is None for t, _, p in rows)
        print(f"[metric] {scenario.name}: {frag!r} is a root: {'yes' if is_root else 'NO'}")
    for frag in scenario.metric_contains:
        print(f"[metric] {scenario.name}: context contains {frag!r}: {'yes' if frag in low else 'NO'}")
    task_src = [s for s in sources if s["type"] == "task"]
    depth = task_src[0]["details"].get("depth") if task_src else None
    print(f"[metric] {scenario.name}: active task depth={depth}")
    proactive_from_task = [
        s for s in sources
        if s["type"].startswith("proactive_") and s["details"].get("path")
        and str(s["details"]["path"][0][0]).startswith("Task:")
    ]
    print(f"[metric] {scenario.name}: proactive items carried by a Task seed: {len(proactive_from_task)}")
```

- [ ] **Step 3: Run once, read the output**

Run: `.venv/bin/python -m pytest tests/agentic/test_goal_hierarchy.py -v -s 2>&1 | tee /private/tmp/claude-501/-Users-ayankawlekar-Desktop-desktop-coding-ideas-github-projects-MemCache/8203cf1e-0b46-439b-abd8-3fa520d98337/scratchpad/hierarchy_run1.log`
Expected: 4 tests run; gates pass; `[tree]`/`[context]`/`[metric]` blocks printed. Read every context qualitatively: does S3 in `top_down` see the alembic failure *and* the umbrella? Does `unrelated_stays_flat` keep two roots?

- [ ] **Step 4: Run the trial series for gate promotion**

Run the file **five** times, saving each log (`hierarchy_run2.log` … `hierarchy_run5.log`). Tabulate per scenario: gate pass/fail; each `[metric]` yes/NO. Decide by the spec's rule:
- If `top_down`'s `[metric] edge 'duplicate' -> 'migrat'` and `'migrat' -> 'telemetry v2'` and `context contains 'under:'` are `yes` on **all** trials, promote `"under:"` (and `"telemetry v2"`) from `metric_contains` to `gate_contains` for `top_down` and `bottom_up`.
- If any trial says NO, they stay metrics. Record the tally either way in Task 11's addendum.
- If `unrelated_stays_flat` ever shows a `SUBGOAL_OF` between telemetry and offsite, that is a precision failure: set `task_placement_min_score` to the smallest value that separates the two in `[metric]` output (print `placement_score` for the pair from a scratch script against the live graph) and re-run.

- [ ] **Step 5: Commit**

```bash
git add tests/agentic/hierarchy_scenarios.py tests/agentic/test_goal_hierarchy.py
git commit -m "test: agentic goal-hierarchy scenarios — top-down, bottom-up, unrelated, siblings

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Calibration, docs, full-suite verification

**Files:**
- Modify: `scripts/calibrate_activation.py` (add a `--task <id>` mode)
- Modify: `steps taken/l4_obstacles_and_decisions.md` (addendum §13–§15)
- Modify: `README.md` (diagram + paragraph)
- Modify: `docs/superpowers/specs/2026-08-19-goal-hierarchy-design.md` (record the one-line `Current task:` deviation from §4a)

- [ ] **Step 1: Extend the calibration script**

In `scripts/calibrate_activation.py`, after computing `seeds_in`, add a Task-seed mode:

```python
    task_ids = [a.split("task:", 1)[1] for a in sys.argv[1:] if a.startswith("task:")]
    for tid in task_ids:
        from app.services.task_store import TaskStore
        from app.services.proactive import lineage_task_seeds
        lineage = TaskStore(driver).get_lineage_ids(tid)
        seeds = lineage_task_seeds(lineage, base=settings.proactive_task_node_seed,
                                   decay=settings.proactive_task_depth_decay)
        nb = g.fetch_neighborhood([], task_ids=lineage, radius=settings.proactive_fetch_radius)
        act = spread_activation(nb, seeds=seeds, floor=settings.proactive_activation_floor,
                                decay=settings.proactive_decay_per_hop).scores
        by_kind: dict[str, list[float]] = defaultdict(list)
        for nid, a in act.items():
            by_kind[nid.split(":", 1)[0]].append(a)
        print(f"== task lineage {lineage}: {len(nb.edges)} edges, {len(act)} nodes")
        for kind, vals in sorted(by_kind.items(), key=lambda kv: -max(kv[1])):
            vals.sort(reverse=True)
            print(f"   {kind:12} n={len(vals):3}  max={vals[0]:.3f}  median={vals[len(vals)//2]:.3f}  min={vals[-1]:.3f}")
```

and make the entity-seed loop skip `task:`-prefixed args. Run it against a leaf task id left in the graph after an agentic run (query one: `MATCH (t:Task)-[:SUBGOAL_OF]->() RETURN t.id LIMIT 1`). Record the printed numbers.

- [ ] **Step 2: Obstacles addendum**

Append to `steps taken/l4_obstacles_and_decisions.md`:

```markdown
## Goal hierarchy (2026-08-19)

### 13. Ancestor entity seeds land at the floor; seed the Task nodes instead

First draft seeded ancestors' *entities* decayed by depth. Working the
arithmetic by hand: `MENTIONS` is counted, so at count 1 it carries 0.205, and
a parent's tool call landed at 0.0496 against a 0.05 floor. The feature would
have worked by luck. Seeding the lineage `Task` nodes at 0.2·0.7^d rides
all-structural edges (0.9 each): leaf tool calls 0.104, parent 0.073,
grandparent 0.051, depth 3 dies; live evidence (0.164) still outranks the
goal's own history (0.144). Pinned by `test_spec_4c_arithmetic_is_pinned`.
Measured live with `calibrate_activation.py task:<leaf-id>`: <PASTE NUMBERS>.

### 14. Roots-only candidates would cap the tree at depth 2

A subgoal of a subgoal needs the mid-level task as a `child_of` candidate, and
that task is not a root. Candidates are now every open task minus the subject's
own subtree; only `parent_of` requires a root. Cycle safety follows from the
exclusion, and `set_parent` re-checks with one bounded descendant walk.

### 15. Gate-promotion tally for the agentic hierarchy scenarios

Five trials of `tests/agentic/test_goal_hierarchy.py` (qwen2.5:3b, temp 0):

| scenario | gate | edge metrics (yes/5) | `under:` in context (yes/5) |
|----------|------|----------------------|-----------------------------|
| top_down | <n>/5 | duplicate→migrat <n>, migrat→telemetry v2 <n> | <n> |
| bottom_up | <n>/5 | duplicate→migrat <n> | <n> |
| unrelated_stays_flat | <n>/5 | (roots: telemetry <n>, offsite <n>) | — |
| sibling_subgoals | <n>/5 | migrat→v2 <n>, grafana→v2 <n> | <n> |

Decision: <promoted "under:" to a gate for … / kept as metric because …>.
```

Fill every `<…>` from the logs before committing. **No placeholders may remain.**

- [ ] **Step 3: README**

In the graph diagram, add the self-edge line under `(:Task)`:

```
(:UserProfile)──PURSUES──▶(:Task)──SUBGOAL_OF──▶(:Task)
```

and after the "Retrieval is proactive" paragraph add:

```markdown
**Goals form a tree.** A second, single-question adjudication decides whether a
new goal is a step toward an existing one (`SUBGOAL_OF`), or the other way
round — precision-first, ≤3 structurally shortlisted candidates, anything
ambiguous resolves to no edge. Retrieval then works up the lineage: the
`Current task:` line reads `Fix duplicate column (under: Migrate schema ▸ Ship
telemetry v2)`, Known Failures ranks the leaf's failures then its ancestors',
and the lineage Task nodes seed activation so a parent goal's failing tool call
surfaces in a fresh session with the path that carried it.
```

Update the test count in "Testing" after Step 4.

- [ ] **Step 4: Full suite, three times**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -5` — three consecutive runs. All must be green. Record the final count.

- [ ] **Step 5: Spec deviation note**

In the spec §4a, replace "Rendered as one line when there are no ancestors, two when there are." with "Rendered as **one** line always — the section merger pairs lines and sources 1:1." (This was found in implementation; the spec records it.)

- [ ] **Step 6: Commit**

```bash
git add -f scripts/calibrate_activation.py "steps taken/l4_obstacles_and_decisions.md" README.md docs/superpowers/specs/2026-08-19-goal-hierarchy-design.md
git commit -m "docs: hierarchy calibration, gate-promotion tally, README; calibrate script task mode

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- §1 schema/invariants → Tasks 2, 3. `last_advanced_at` backfill via coalesce → Task 3.
- §2 store API → Tasks 2, 3, 4. Every method named in the spec table has a task.
- §3a shortlist / §3b parse+call / §3c worker → Tasks 5, 6.
- §4a path line / §4b failures / §4c Task seeds + `SUBGOAL_OF` prior → Tasks 1, 7, 8, 9. §4a's "one or two lines" corrected to one (Task 11 step 5).
- §5 config → Task 1.
- §6 failure handling → Task 6 (`TaskHierarchyError` test), Task 9 (degrade paths inherit existing try/except).
- §7 deterministic tests → Tasks 1–9; severed tests → Task 9; agentic → Task 10; gate-promotion → Task 10 step 4 + Task 11 §15.
- §8 scope → nothing out-of-scope is built.
- Names consistent: `set_parent/get_parent/get_ancestors/get_children/get_descendant_ids/get_lineage_ids/active_task/list_placement_candidates/task_evidence`, `PlacementCandidate`, `PlacementVerdict`, `parse_placement/build_placement_prompt/shortlist_candidates/placement_score/adjudicate_placement`, `lineage_task_seeds`, `build_seeds(task_nodes=)`, `fetch_neighborhood(task_ids=)`, `failed_calls(task_ids=)`, `_write_hierarchy`.
