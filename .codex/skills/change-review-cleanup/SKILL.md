---
name: change-review-cleanup
description: "Use when Codex needs to review and organize an existing dirty worktree before commit: classify current changes, remove obsolete test code and unused traces, add or refine .gitignore rules for generated experiment artifacts, record key experiment commands/results in Markdown, remove debug dumps or development-only instrumentation from production-coupled code, perform scoped quality refactors when useful, update feature or top-level README documentation for significant functionality, and prepare categorized commits only after explicit user approval."
---

# Change Review Cleanup

Use this skill to turn an exploratory worktree into a reviewable, documented, commit-ready change set. Optimize for preserving useful work, deleting obvious dead artifacts, and making the remaining diff easy for a human to review.

## Core Rules

1. Never revert or delete changes that may belong to the user unless the user explicitly asked for that exact cleanup or the files are clearly generated disposable artifacts.
2. Do not commit until you have shown the user the categorized commit plan and received explicit approval.
3. Do not push unless the user explicitly gives a branch/remote instruction or confirms the proposed push target.
4. Keep cleanup scoped to the active request. Avoid opportunistic rewrites outside the touched feature area.
5. Prefer reproducible scripts and Markdown summaries over checking in bulky generated experiment outputs.

## Workflow

### 1. Inventory the Worktree

Start by collecting a high-signal map:

- `git status --short`
- `git diff --stat`
- targeted `git diff -- <path>` for changed source files
- `git ls-files --others --exclude-standard` for untracked files
- `rg --files` when searching for matching traces, outputs, reports, debug dumps, or generated data

Classify every relevant path into one of these buckets:

- **production source**: runtime/library code that ships or is imported by shipped paths
- **experiment source**: scripts, simulators, generators, notebooks, benchmark harnesses
- **tests**: unit/integration tests that should remain meaningful after cleanup
- **docs**: worklogs, result summaries, README updates, feature docs
- **canonical inputs**: small traces/configs/profiles needed to reproduce results
- **generated outputs**: logs, plots, SVG/PNG timelines, JSON summaries, caches, model outputs, temp dumps
- **obsolete artifacts**: abandoned traces, one-off debugging scripts, stale tests, empty outputs

If ownership or intent is unclear, inspect the file before classifying it.

### 2. Remove Obsolete Tests and Traces

Delete only when one of these is true:

- The file was created for a discarded approach and is not referenced by docs, scripts, tests, or README.
- A newer canonical trace supersedes it and the older trace adds no coverage or reproducibility value.
- A test covers a behavior that no longer exists and cannot be updated into a useful invariant.
- The file is a generated output that can be recreated by a checked-in command.

Prefer keeping a minimal set:

- one or two canonical traces per important workload shape
- one smoke trace for fast local checks
- one realistic trace for strategy evaluation
- tests that protect invariants, bugs fixed during the work, or public behavior

Before deleting, use `rg` to check references. After deleting, rerun affected tests or scripts.

### 3. Ignore Generated Experiment Data

Use `.gitignore` or a feature-local ignore file to cover generated outputs, not source inputs.

Common generated artifacts to ignore:

- experiment `out/` directories
- raw logs and command captures
- generated plots/timelines/contact sheets
- benchmark JSON/CSV summaries that are reproducible and bulky
- temp debug dumps, caches, and partial downloads

Do not ignore:

- source scripts that generate the data
- small canonical input traces/configs/profiles
- Markdown summaries that record key conclusions
- golden fixtures required by tests

When adding ignore rules, include comments that explain the artifact family if the pattern is not obvious.

### 4. Record Key Experiments in Markdown

For every important experiment or benchmark, keep a concise Markdown record close to the feature:

- objective and date
- exact command(s) or script entry point
- important inputs such as trace names, model/cost model, GPU count, SLO thresholds
- key metrics and conclusions
- known limitations and what the results should not be used for
- links or paths to regenerated outputs when useful

Prefer a durable `WORKLOG.md`, `results.md`, or feature README over scattered notes. Generated images or JSON can stay ignored if the Markdown records how to recreate them.

### 5. Clean Production-Coupled Code

Review any code imported by the real system, not only experiment scripts. Remove or tighten:

- unconditional `print`, `pdb`, `breakpoint`, debug `assert False`, or noisy ad hoc logging
- debug dump paths, temporary JSON/PNG saves, and local absolute paths
- environment flags introduced only for manual experiments
- broad exception swallowing used during exploration
- one-off instrumentation that changes behavior, synchronization, memory use, or latency

If observability is genuinely useful, convert it to production-appropriate logging or metrics behind an existing config flag. Keep runtime APIs stable unless the user asked for a breaking change.

### 6. Refactor When It Reduces Review Risk

Refactor only when it makes the remaining change easier to maintain or safer to test:

- extract duplicated experiment parsing or rendering helpers
- name scheduler/cost-model concepts consistently
- isolate generated-output writing from simulation logic
- split large scripts only when the boundary is already clear
- replace ad hoc parsing with structured parsers when the input format allows it

Avoid broad formatting churn, unrelated style changes, or speculative abstractions.

### 7. Update README and Feature Docs

If the work creates a key reusable feature, update the appropriate docs:

- feature-level README for commands, inputs, outputs, and limitations
- top-level README only for user-visible or project-level capabilities
- experiment docs for baselines, trace semantics, and reproducibility notes

Do not advertise scaffolding as production-ready. Name prototype status, unsupported paths, and validation gaps.

### 8. Validate

Choose validation proportional to the diff:

- `git diff --check` on touched files
- `python -m py_compile` for changed Python scripts when full tests are expensive
- targeted unit tests for production code and simulator invariants
- regeneration commands for Markdown summaries or plots that changed
- manual inspection of generated Markdown/SVG/index files when visualization output changed

Report skipped validation explicitly with the reason.

### 9. Prepare Commit Plan and Stop

Before committing, present the user with:

- files to keep, grouped by category
- files deleted or intentionally left untracked
- ignore rules added or changed
- docs updated
- tests/commands run
- proposed commit split and commit messages
- any remaining risks or decisions requiring user input

Ask for approval to commit. After approval, create categorized commits with non-interactive git commands. Push only after a separate explicit user instruction or approval of a proposed remote/branch.

## Acceptance Checklist

Before finalizing a cleanup pass:

- obsolete traces/tests/generated artifacts are removed or ignored
- canonical inputs and scripts needed for reproduction remain tracked
- key experiment conclusions are recorded in Markdown
- production-coupled code has no stray debug dumps or development-only instrumentation
- docs mention significant new features and current limitations
- validation has been run and summarized
- commit has not been made without explicit user approval
