# Snapshot Time Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align automatic snapshot states with model time by rotating time indices between diagnostic sampling and output.

**Architecture:** Keep physics, plugin execution, clock advancement, `after_timestep`, and diagnostic sampling in their existing order. Rotate once after sampling, then let every diagnostic output read the completed state through the normal `tau` convention. Remove the branch's optional slot-selection interfaces and tendency exceptions.

**Tech Stack:** Python, NumPy, pytest, h5netcdf, Veros diagnostics.

**Spec:** The recommendation in this conversation, approved by the user's request to implement it. PyOM's batch drivers diagnose before rotation using the old clock; Veros advances its clock earlier, so output must follow rotation.

## Global Constraints

- Do not change any physics kernel, numerical integration formula, timestep size, or tendency storage convention.
- Preserve the pre-rotation state seen by plugins, `after_timestep`, energy sampling, averages, and overturning sampling.
- Keep `output(state)` compatible with existing custom Snapshot subclasses.
- Output all variables with a model timestep dimension from `tau`, including raw tendencies; the latest standard tendencies are at `taum1` after rotation, whereas the latest `dpsi` is at `tau`.
- Work in the existing `fix/snapshot-time-alignment` checkout, as requested. The subsequent user request authorizes committing and pushing the reviewed changes to `origin`; no GitHub messages are requested.

## Review Focus

- Rotation before sampling corrupts two-slot energy differences and shifts averages: cover analytic averages and a physical ACC trajectory with all energy components enabled.
- Rotating before `after_timestep` changes setup hooks: cover the input/completed tracer values observed by the hook.
- Output-only diagnostics must now see the completed state: cover a custom Snapshot and the tracer monitor.
- Sparse output and all three slot permutations must agree with manual output: cover both output frequencies and sentinel slots, including custom states and raw tendencies.
- Restarting after rotation must preserve trajectories and accumulated diagnostics: run the existing restart regression and compare baseline ACC state and diagnostic output over six steps with unequal momentum/tracer timesteps.

---

### Task 1: Rotate between sampling and output

**Files:**
- Modify: `veros/veros.py`
- Restore original writer interfaces: `veros/diagnostics/api.py`, `veros/diagnostics/base.py`, `veros/diagnostics/snapshot.py`, `veros/io_tools/netcdf.py`
- Modify: `doc/reference/diagnostics.rst`
- Test: `test/snapshot_test.py`, existing `test/energy_test.py`, `test/restart_test.py`

**Interfaces:**
- Consumes: `diagnostics.diagnose(state)` with input state at `tau` and completed state at `taup1`.
- Produces: `diagnostics.output(state)` with completed state at `tau`; standard `Snapshot.output(state)`, `VerosDiagnostic.write_output(state)`, and `write_variable(..., time_step=-1)` signatures.

- [x] **Step 1: Capture the clean baseline.** Run the full NumPy suite and capture six ACC steps, all model variables, and energy/averages/overturning output with TKE, EKE, IDEMIX enabled, for streamfunction and pressure solvers.

  Run: `VEROS_ASSET_DIR=/tmp/veros-test-assets .venv/bin/python -m pytest --forked -q`
  Expected: green suite; baseline captures contain finite physical state and diagnostic values.

- [x] **Step 2: Write regression tests before changing production code.** Keep the forced-box analytic temperature/salinity test. Exercise custom `Snapshot.output(state)`, analytic input-state averages, unchanged `after_timestep` observations, output-only tracer monitoring, and uniform slot selection for all built-in and custom timestep variables across three permutations.

  Run: `.venv/bin/python -m pytest --forked -q test/snapshot_test.py test/energy_test.py`
  Expected: new automatic-output checks fail against the branch's pre-rotation behavior, while existing analytic snapshot coverage stays valid.

- [x] **Step 3: Make the minimal production change and update documentation.** Move the existing rotation immediately after `diagnostics.diagnose(state)`, use `diagnostics.output(state)`, restore the four writer files to their pre-branch interfaces, and document sampling versus output stages and raw tendency semantics.

  Run: `.venv/bin/python -m pytest --forked -q test/snapshot_test.py test/energy_test.py test/restart_test.py`
  Expected: all targeted tests pass.

- [x] **Step 4: Verify physics and the final change.** Compare all six post-step ACC states and the energy/averages/overturning files against the captured baseline, require exact equality, run the full NumPy suite, and check formatting and whitespace.

  Run: `VEROS_ASSET_DIR=/tmp/veros-test-assets .venv/bin/python -m pytest --forked -q`
  Expected: green suite; unchanged model trajectories and sampled diagnostics. Record unavailable backend or PyOM comparison capabilities explicitly.

- [x] **Step 5: Request an independent review.** Review the working-tree diff and the net branch diff against the constraints above, address substantive findings, and leave the reviewed changes in the requested checkout.

## Verification Results

- Baseline NumPy suite: 86 passed, 71 skipped.
- New regression tests before implementation: 3 expected failures (custom Snapshot interface and stale tracer-monitor state), 8 passed.
- Snapshot, energy, and restart regressions after implementation: 12 passed.
- Final full NumPy suite: 91 passed, 71 skipped.
- Six ACC timesteps with unequal momentum/tracer timesteps, TKE/EKE/IDEMIX enabled: all 886 streamfunction and 823 pressure arrays exactly match the baseline, including model variables and energy/averages/overturning file contents.
- In-memory mutations on either side of the rotation boundary trigger the expected regression failures: early rotation breaks energy/averages; late rotation breaks snapshots/tracer monitoring.
- Black and whitespace checks passed. Independent review found no issues.
- JAX, MPI, and a compiled PyOM library are unavailable locally; those checks were not run.
- The user subsequently requested committing and pushing the reviewed changes to the existing remote branch using GitHub CLI authentication.
