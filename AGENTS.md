# AGENTS.md — chancewareVI contributor guide

chancewareVI is an Xbox Series X|S compatibility layer for Windows. These rules apply to people and agents alike. Keep `main` usable: make each change easy to review, verify, and revert.

## Keep title research local

- Use only title files you are authorized to use. Select your own executable and folder at runtime; never hard-code a local path or assume another contributor has the same files.
- Keep title and publisher names, build fingerprints, executable offsets, symbols, decompilation, disassembly, and notes about specific titles private. Keep binaries, assets, saves, shader dumps, screenshots, traces, and raw logs outside the repository. Do not upload them to GitHub, hosted AI tools, or shared agent memory.
- Publish only redistributable compatibility code and original synthetic fixtures. Explain issues in platform terms such as descriptor bounds, texture layout, or wave behavior. Use sanitized symptoms and measurements instead of private evidence.

## Work from a current base

Start in a clean checkout of the public chancewareVI repository or your fork. Use `origin` for a direct clone; for a fork, use `upstream` pointing at chancewareVI and `origin` pointing at your fork. Check `git remote -v` before fetching or pushing. Do not base public work on private development history or another compatibility-layer fork.

```powershell
git status --short --branch
git remote -v
$base = 'origin' # Use 'upstream' in a contributor fork.
git fetch $base
git switch --no-track -c fix/descriptor-bounds "${base}/main"
cargo xtask install-hooks
```

Use one short-lived branch per issue or coherent change. Name it `fix/…`, `feat/…`, or `docs/…`. If unrelated work is in progress, use a separate worktree rather than stashing, overwriting, or mixing it with the new change.

Check existing issues and PRs before taking on overlapping work. For work that touches a shared subsystem or changes a public contract, record the scope, owner, acceptance checks, and dependencies in the issue or draft PR. Coordinate changes to the same code and identify work that must land first.

Fetch the base before marking the PR ready. Rebase only your own topic branch; coordinate first on shared branches. If you must update a rebased personal branch, use `--force-with-lease` only on that branch. Never force-push `main` or another contributor's branch.

## Make and verify the change

- Keep changes small and focused. Reuse existing code and platform APIs; avoid drive-by refactors and optional dependencies.
- Use the toolchain pinned in `rust-toolchain.toml`. Run `cargo xtask policy` and the narrowest tests that cover the change. A fix should fail before and pass after under a reproducible check where practical. Use synthetic inputs in tests that run for everyone.
- For rendering and performance changes, include the [device specifications](#device-specifications-for-runtime-reports) and observed output. Compare the same workload and settings before and after; check that the output still matches. If an adapter or workload is untested, say so rather than claiming parity. A successful build or pipeline creation does not establish correct rendering.
- Keep unrelated changes out of the PR. Do not loosen tests, validation, or publication checks to get a green result. When a regression is found, link the introducing change and coordinate a focused fix or revert before stacking dependent work on top.
- Suggested tools, not requirements: **Ponytail** for small, complete changes; **Caveman** for concise updates without losing technical qualifiers; **Graphify** for navigating source relationships; **REA** or **IDA Pro MCP** for authorized local binary analysis. Keep any graphs or databases containing private inputs local.

## Triage issues and crashes

1. Search for a duplicate by subsystem, failure phase, and sanitized error code. In a new issue, record the chancewareVI revision, device specifications, expected and actual behavior, and the shortest reproduction that does not identify the title. Include the last known good revision if you have one. If you cannot reproduce it, say what you checked.
2. Reproduce locally with an unmodified diagnostics build and the same inputs. Start with the normal runtime log. Inspect `crashes.log` in the title's local runtime data directory if present. Identify the earliest failure and its phase: setup/import binding, CPU exception, GPU validation/device removal, or rendering without a crash. For GPU failures, rerun with the D3D12 debug layer; add DRED for device removal. Use the [README diagnostics commands](README.md#testing). Keep full logs, symbols, offsets, and debugger output private.
3. Link the fix PR to the issue. Identify the violated invariant, prove the smallest fix with a regression check, and report which revisions, GPUs, and modes were exercised. Close the issue when the fix is merged and verified; leave unverified variants open and record what evidence is missing.

If a report or commit exposes credentials or private material, stop publication, notify a maintainer through a private channel, and revoke or rotate exposed credentials. Deleting an issue attachment or making a follow-up commit may not remove cached data or prior Git objects.

## Device specifications for runtime reports

Include this information in runtime issues and PRs with hardware test results. Report the adapter that actually ran the workload, especially on laptops with integrated and discrete GPUs.

- **System:** Windows edition and OS build, CPU model, installed RAM, and desktop/laptop/VM. For streaming or I/O problems, include the storage type (for example, NVMe SSD).
- **Graphics:** GPU model and vendor, dedicated VRAM if applicable, and exact graphics driver version. For multiple GPUs, list the others and identify the active one. State whether the run used physical hardware or WARP.
- **Run:** chancewareVI commit or release, build profile, console mode, normal or Experimental RT mode, and non-default settings. For rendering issues, include resolution and HDR state.

Use `winver`, Windows Settings, or `dxdiag` locally to look up individual fields. Write `unknown` for anything you cannot confirm. Copy only the values above; do not attach full system reports or include usernames, machine names, device serials, or account identifiers.

## Before every push or pull request

1. Confirm the destination and inspect the full set of commits to be sent, not just the latest diff. Review files, commit messages, PR text, and attachments for private material, embedded data, and secrets. Do not push if an earlier outgoing commit contains excluded material. Never mirror private development refs or history.
2. Preserve `LICENSE`, upstream credits, copyright notices, and third-party terms. Check redistribution rights for every added dependency or imported file. Do not publish build artifacts containing separately licensed inputs without clearance and the corresponding source required by the license.
3. Open a pull request for changes to the public default branch. Include what changed, the checks run and their results, known limits, and your AI disclosure. Require a human review before merge; a passing policy check does not establish redistribution rights.

```powershell
git log --stat --oneline "${base}/main..HEAD"
git diff --check "${base}/main...HEAD"
cargo xtask policy
```

Read the outgoing commits and diff, not just the command output. After the review passes, run `git push -u origin HEAD` and open the PR against chancewareVI `main`. Make sure `origin` is the intended push destination.

Before the first public commit, check `git var GIT_AUTHOR_IDENT`. If you want to keep a personal email address private, configure your GitHub-provided no-reply address for this checkout before committing. Do not rewrite other authors' identities or credits. Ignore rules do not hide personal information already committed.

## Review and merge

- The author records the change, how to verify it, what was not tested, and any migration or rollback risk. Keep follow-up work separate and link it when it matters to the change.
- A human reviewer checks correctness, compatibility, regressions, licensing, and data exposure. Resolve review feedback, rerun affected checks after edits, and update the PR with the new evidence.
- On GitHub, configure a ruleset for `main` requiring pull requests, human approval, and applicable CI checks. Enable secret scanning and push protection when available. Do not bypass required checks or an unresolved review. Merge only when the stated acceptance checks are met.

## AI disclosure

Include an `AI assistance` section in each PR. If none was used, say `AI assistance: none`. Otherwise list the tool or client, exact model ID or version shown by the tool (use `not shown` if unavailable), and what it did. Include material contributions from additional models or agents. State what a human reviewed and which checks actually ran. Do not include prompts, transcripts, or private inputs.
