"""Executable research acceptance experiments for Veros differentiation."""

from .matrix import CASES, RESPONSES, TASKS

DEMOS = {
    "bathymetry_optimization": "Partial-cell sill-depth recovery, held-out trajectories, and checkpoint gradients",
    "forcing_response": "Budget-balanced forcing sensitivities and a constrained finite update",
    "state_forcing_twin": "Joint initial-state and forcing estimation with held-out observations",
    "mixing_calibration": "Multi-diagnostic mixing calibration and longer-rollout validation",
    "observation_design": "Control identifiability and observation selection",
    "closure_learning": "Positive state-dependent column diffusion and held-out rollouts",
    "equilibrium_sensitivity": "Implicit column equilibrium sensitivity versus converged unrolling",
    "gradient_horizon": "Derivative agreement across increasing ocean trajectory lengths",
    "checkpoint_replay": "Objective and control-gradient agreement under RAM/disk replay",
}

DEMOS.update({name: f"{TASKS[case.task]}: {RESPONSES[case.response]}" for name, case in CASES.items()})
