"""Scientific coverage and executable contracts for the optimization matrix."""

from dataclasses import dataclass

TASKS = {
    "sensitivity": "Sensitivity and mechanism analysis",
    "estimation": "State and forcing estimation",
    "calibration": "Physical parameter calibration",
    "perturbation": "Optimal perturbations and constrained experiments",
    "closure": "Learning parameterizations",
    "observation": "Identifiability, uncertainty, and observing-system design",
    "geometry": "Geometry and discretization design",
}

RESPONSES = {
    "local": "Local tendency or one-step response",
    "trajectory": "Finite trajectory",
    "equilibrium": "Stable equilibrium",
    "seasonal": "Adjusted seasonal cycle",
    "statistics": "Long-term climate statistics",
}

# Stable question IDs link experiments to coverage checks.
QUESTIONS = {
    "local_forcing": "Which heat, salinity, and wind inputs influence immediate regional tracer and transport responses?",
    "local_calibration": "Can local tracer and momentum responses distinguish mixing coefficients from wind stress?",
    "steady_forcing": "Which steady forcing corrections reconcile adjusted hydrography and transports?",
    "regional_heat": "Which regional heat-flux anomalies affect North Atlantic upper-ocean heat content?",
    "southern_wind": "How does sensitivity to Southern Ocean winds change with response time and season?",
    "initial_vs_forcing": "Can T/S observations distinguish initial stratification errors from surface-flux errors?",
    "seasonal_forcing": "Which seasonal heat, wind, and salinity-flux corrections reconcile basin observations?",
    "isopycnal_transport": "Can isopycnal mixing improve thermocline structure without worsening heat transport?",
    "wind_vs_mixing": "Can observations distinguish wind-stress errors from vertical and isopycnal mixing errors?",
    "balanced_heat": "Which budget-balanced forcing redistribution maximizes regional heat content?",
    "overturning_target": "What minimum-cost buoyancy perturbation reduces overturning by 10% within ten years?",
    "reference_tendencies": "Can a positive state-dependent diffusivity reproduce coarse-grained reference tendencies?",
    "closure_transfer": "Does a tendency-trained closure improve withheld seasonal stratification and circulation?",
    "observable_controls": "Which combinations of forcing, initial state, and mixing are observable?",
    "transport_observations": "Which T/S sensors reduce uncertainty in future transport?",
    "sill_exchange": "How does sill depth influence exchange?",
    "basin_bathymetry": "Which admissible bathymetric modification changes basin transport most efficiently?",
    "vertical_grid": "Can vertical-grid placement improve accuracy at fixed resolution?",
    "topology": "Can a channel-opening design retain its response when realized as a discrete wet-cell geometry?",
    "climatological_fit": "Can hydrography, heat transport, overturning, and ACC transport be calibrated together?",
    "climate_response": "How do forcing, mixing, closures, and geometry change climate means, variability, and extremes?",
}

TASK_QUESTIONS = {
    "sensitivity": ("regional_heat", "southern_wind"),
    "estimation": ("initial_vs_forcing", "seasonal_forcing"),
    "calibration": ("isopycnal_transport", "wind_vs_mixing", "climatological_fit"),
    "perturbation": ("balanced_heat", "overturning_target"),
    "closure": ("reference_tendencies", "closure_transfer"),
    "observation": ("observable_controls", "transport_observations", "wind_vs_mixing"),
    "geometry": ("sill_exchange", "basin_bathymetry", "vertical_grid", "topology"),
}

CONTRACTS = {
    "equilibrium_response": "A JAX-transformable fixed-point response, including initial-state adjustment, with a scaled full-state residual and stability estimate on the admissible prognostic tangent space. Fix conserved inventories and solver gauges, and exclude the clock. Steady forcing is obtained by holding the annual phase at zero.",
    "periodic_response": "A JAX-transformable 360-day Poincare response, including the derivative of the phase-zero state. Return phase-resolved observations, a scaled full-state cycle residual, and stability estimate on the admissible prognostic tangent space with conserved inventories, solver gauges, and phase fixed.",
    "statistical_response": "A JAX-transformable invariant-statistics response for means, variances, and smooth exceedance frequencies. Return standard errors, effective sample size, split-window drift, and method metadata; validate against independent long forward ensembles.",
    "with_tracer_closure": "Install a pure JAX conservative vertical tracer-flux law with dynamic trainable weights. It must act inside every timestep and preserve carry/checkpoint differentiation.",
    "reference_tracer_tendencies": "Return coarse-grid tracer tendencies from an independent higher-resolution Veros reference, conservatively projected to the experiment grid, plus closure-only heat/salt budget residuals. No reference generated by the fitted closure.",
    "with_vertical_grid": "Rebuild vertical coordinates, metrics, density, masks, pressure operators, and solver coefficients inside AD. Conservatively remap prognostic fields and histories to the new fixed-count levels.",
    "with_wet_geometry": "Apply a continuous channel-opening geometry, rebuild masks and all geometry-dependent operators, and conservatively remap state. Also support a realized discrete geometry for forward validation.",
    "grid_reference_error": "Return a tracer/transport error against an independently refined Veros solution, conservatively mapped to the candidate grid. This is the accuracy objective for grid design. Statistical objectives must also support forward=True and seed for independent ensemble validation of invariant derivatives.",
}


@dataclass(frozen=True)
class Case:
    task: str
    response: str

    @property
    def name(self):
        return f"{self.task}_{self.response}"

    @property
    def questions(self):
        questions = TASK_QUESTIONS[self.task]
        if self.task == "sensitivity" and self.response == "local":
            questions = ("local_forcing",)
        if self.task == "calibration" and self.response == "local":
            questions = ("local_calibration", "wind_vs_mixing")
        if self.task == "calibration" and self.response == "trajectory":
            questions = ("isopycnal_transport", "wind_vs_mixing")
        if self.task == "estimation" and self.response == "equilibrium":
            questions = ("steady_forcing",)
        if self.task == "estimation" and self.response in ("seasonal", "statistics"):
            questions = ("seasonal_forcing",)
        if self.task == "perturbation" and self.response != "trajectory":
            questions = ("balanced_heat",)
        if self.response == "statistics":
            questions += ("climate_response",)
        return questions

    @property
    def requirements(self):
        methods = {
            "equilibrium": ("equilibrium_response",),
            "seasonal": ("periodic_response",),
            "statistics": ("statistical_response",),
        }.get(self.response, ())
        if self.task == "closure":
            methods += ("with_tracer_closure", "reference_tracer_tendencies")
        if self.task == "geometry":
            methods += ("with_vertical_grid", "with_wet_geometry", "grid_reference_error")
        return methods


CASES = {case.name: case for task in TASKS for response in RESPONSES for case in (Case(task, response),)}


def manifest():
    return dict(
        tasks=TASKS,
        responses=RESPONSES,
        questions=QUESTIONS,
        cases={
            name: dict(
                task=case.task, response=case.response, questions=case.questions, requires=case.requirements
            )
            for name, case in CASES.items()
        },
        contracts=CONTRACTS,
    )


def missing_capabilities(prepared_type, case):
    return {
        method: CONTRACTS[method]
        for method in case.requirements
        if not callable(getattr(prepared_type, method, None))
    }
