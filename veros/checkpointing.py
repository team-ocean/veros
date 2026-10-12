"""Host-managed checkpoint schedules, storage, and stepwise reverse integration."""

from collections import namedtuple
import math
import inspect
import operator
from pathlib import Path
import tempfile
from contextlib import ExitStack

CheckpointAction = namedtuple("CheckpointAction", "kind index end storage", defaults=(None, None))
CheckpointAction.__doc__ = """A schedule action at a timestep index.

``advance`` and ``reverse`` use the exclusive ``end`` index. ``save`` uses
``storage='ram'`` or ``storage='disk'``. Other actions only use ``index``.
"""
CheckpointedResult = namedtuple(
    "CheckpointedResult", "value gradient final statistics parameter_gradient", defaults=(None,)
)
CheckpointedResult.__doc__ = "Loss, initial-carry cotangent, final carry, storage statistics, and setting gradients."


def _integer(value, name, minimum=0):
    value = operator.index(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _split(length, extra):
    # Binomial capacity for the segment's starting checkpoint and extra slots.
    lower, upper = 0, 1
    while math.comb(extra + upper + 1, extra + 1) < length:
        upper *= 2
    while lower < upper:
        middle = (lower + upper) // 2
        if math.comb(extra + middle + 1, extra + 1) < length:
            lower = middle + 1
        else:
            upper = middle
    return min(length - 1, math.comb(extra + lower, extra + 1))


def _materialize(start, end, extra, disk=0):
    while extra and end - start > 1:
        middle = start + _split(end - start, extra)
        yield CheckpointAction("advance", start, middle)
        yield CheckpointAction("save", middle, storage="disk" if disk else "ram")
        start, extra = middle, extra - 1
        disk = max(0, disk - 1)
    if start < end:
        yield CheckpointAction("advance", start, end)


def _reverse(start, end, extra, checkpoints_prepared, disk=0):
    # An explicit stack also handles long horizons without Python recursion.
    tasks = [("segment", start, end, extra, checkpoints_prepared, disk)]
    while tasks:
        kind, start, end, extra, checkpoints_prepared, disk = tasks.pop()
        if kind == "drop":
            yield CheckpointAction("drop", start)
            continue
        if start == end:
            continue
        if not extra or end - start == 1:
            for index in range(end - 1, start - 1, -1):
                yield CheckpointAction("restore", start)
                if start < index:
                    yield CheckpointAction("advance", start, index)
                yield CheckpointAction("reverse", index, index + 1)
            continue
        middle = start + _split(end - start, extra)
        if not checkpoints_prepared:
            yield CheckpointAction("restore", start)
            # Prepare nested right segments without advancing the final timestep.
            current, slots, disk_slots = start, extra, disk
            while slots and end - current > 1:
                split = current + _split(end - current, slots)
                yield CheckpointAction("advance", current, split)
                yield CheckpointAction("save", split, storage="disk" if disk_slots else "ram")
                current, slots = split, slots - 1
                disk_slots = max(0, disk_slots - 1)
        tasks.extend(
            (
                ("segment", start, middle, extra, False, disk),
                ("drop", middle, None, None, None, None),
                ("segment", middle, end, extra - 1, True, max(0, disk - 1)),
            )
        )


class BinomialSchedule:
    """Binomial splits using fixed budgets for host-RAM and disk slots.

    ``n_steps`` counts model timesteps.
    The initial state occupies one slot. Disk slots hold the outer splits,
    followed by RAM slots at deeper splits. This placement does not optimize
    I/O costs. The live state, final state, adjoint, and step-local
    differentiation buffers are outside the slot budgets.
    """

    def __init__(self, n_steps, *, ram_checkpoints=4, disk_checkpoints=0):
        self.n_steps = _integer(n_steps, "n_steps")
        self.ram_checkpoints = _integer(ram_checkpoints, "ram_checkpoints")
        self.disk_checkpoints = _integer(disk_checkpoints, "disk_checkpoints")
        if not self.ram_checkpoints + self.disk_checkpoints:
            raise ValueError("Provide at least one RAM or disk checkpoint slot")

    def actions(self):
        """Generate save/restore/recompute actions, then release every slot."""
        if not self.n_steps:
            yield CheckpointAction("turn", 0)
            return
        extra = self.ram_checkpoints + self.disk_checkpoints - 1
        disk = max(0, self.disk_checkpoints - 1)
        yield CheckpointAction("save", 0, storage="disk" if self.disk_checkpoints else "ram")
        yield from _materialize(0, self.n_steps, extra, disk)
        yield CheckpointAction("turn", self.n_steps)
        yield from _reverse(0, self.n_steps, extra, True, disk)
        yield CheckpointAction("drop", 0)


class PeriodicDiskSchedule:
    """Periodic disk checkpoints with binomial RAM replay in each interval.

    ``disk_interval`` and ``n_steps`` are measured in model timesteps.
    Disk checkpoints do not consume RAM slots. With zero RAM slots, replay
    repeatedly advances from each interval's disk checkpoint.
    """

    def __init__(self, n_steps, *, disk_interval, ram_checkpoints=2):
        self.n_steps = _integer(n_steps, "n_steps")
        self.disk_interval = _integer(disk_interval, "disk_interval", 1)
        self.ram_checkpoints = _integer(ram_checkpoints, "ram_checkpoints")

    def actions(self):
        if not self.n_steps:
            yield CheckpointAction("turn", 0)
            return
        starts = range(0, self.n_steps, self.disk_interval)
        for start in starts:
            end = min(start + self.disk_interval, self.n_steps)
            yield CheckpointAction("save", start, storage="disk")
            if end == self.n_steps:
                yield from _materialize(start, end, self.ram_checkpoints)
            else:
                yield CheckpointAction("advance", start, end)
        yield CheckpointAction("turn", self.n_steps)
        for start in reversed(starts):
            end = min(start + self.disk_interval, self.n_steps)
            yield from _reverse(start, end, self.ram_checkpoints, end == self.n_steps)
            yield CheckpointAction("drop", start)


def validate_schedule(schedule):
    """Validate a replayable action schedule without running model kernels.

    Custom schedules provide ``n_steps``, ``ram_checkpoints``, and a fresh
    ``actions()`` iterator. Actions use timestep indices. ``turn`` ends the
    original forward traversal. Reverse actions consume one timestep at a time.
    """
    length = _integer(schedule.n_steps, "n_steps")
    capacity = _integer(schedule.ram_checkpoints, "ram_checkpoints")
    disk_capacity = getattr(schedule, "disk_checkpoints", None)
    if disk_capacity is not None:
        disk_capacity = _integer(disk_capacity, "disk_checkpoints")
    checkpoints = {}
    tier_counts = dict(ram=0, disk=0)
    current, frontier, reverse_head = 0, 0, length
    turned = False
    statistics = dict(forward_steps=0, recomputed_steps=0, reverse_steps=0)
    for action in schedule.actions():
        index = _integer(action.index, "checkpoint index")
        if index > length:
            raise ValueError("Checkpoint index exceeds schedule length")
        if action.kind == "save":
            if index != current or index in checkpoints:
                raise ValueError("Save must use the current index and an unused checkpoint")
            if action.storage not in ("ram", "disk"):
                raise ValueError("Checkpoint storage must be ram or disk")
            checkpoints[index] = action.storage
            tier_counts[action.storage] += 1
            if tier_counts["ram"] > capacity:
                raise ValueError("Schedule exceeds its RAM checkpoint budget")
            if disk_capacity is not None and tier_counts["disk"] > disk_capacity:
                raise ValueError("Schedule exceeds its disk checkpoint budget")
        elif action.kind in ("restore", "drop"):
            if index not in checkpoints:
                raise ValueError(f"Cannot {action.kind} missing checkpoint {index}")
            if action.kind == "restore":
                current = index
            else:
                tier_counts[checkpoints[index]] -= 1
                del checkpoints[index]
        elif action.kind == "advance":
            end = _integer(action.end, "advance end")
            if index != current or not index < end <= length:
                raise ValueError("Advance must start at the current index and stay within the schedule")
            if not turned and index != frontier:
                raise ValueError("The original forward traversal must advance monotonically")
            statistics["recomputed_steps" if turned else "forward_steps"] += end - index
            current = end
            if not turned:
                frontier = end
        elif action.kind == "turn":
            if turned or index != length or current != length or frontier != length:
                raise ValueError("Turn must occur once at the end of the forward traversal")
            turned = True
        elif action.kind == "reverse":
            if not turned or index != current or action.end != index + 1 or action.end != reverse_head:
                raise ValueError("Reverse must consume the next timestep in backward order")
            reverse_head = index
            statistics["reverse_steps"] += 1
            statistics["recomputed_steps"] += 1
        else:
            raise ValueError(f"Unknown checkpoint action {action.kind}")
    if not turned or reverse_head or checkpoints:
        raise ValueError("Schedule must finish the reverse traversal and release all checkpoints")
    return statistics


class CheckpointStore:
    """Exact PyTree checkpoints in host RAM or temporary NumPy archives.

    Use as a context manager. Array-only archives occupy an owned subdirectory
    of ``directory`` and are removed on exit. Tree structure stays in the process.
    """

    def __init__(self, template, *, ram_checkpoints, directory=None, disk_checkpoints=None, max_ram_bytes=None):
        import jax
        import numpy as np

        self.ram_checkpoints = _integer(ram_checkpoints, "ram_checkpoints")
        self.disk_checkpoints = None if disk_checkpoints is None else _integer(disk_checkpoints, "disk_checkpoints")
        self.directory = directory
        self.max_ram_bytes = None if max_ram_bytes is None else _integer(max_ram_bytes, "max_ram_bytes")
        leaves, self._structure = jax.tree.flatten(template)
        if any(getattr(leaf, "weak_type", False) for leaf in leaves):
            raise TypeError("Checkpoint leaves must have explicit dtypes, as in a packed Veros carry")
        self._schema = [(tuple(leaf.shape), np.dtype(leaf.dtype)) for leaf in leaves]
        self.state_bytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in self._schema)
        if any(dtype.hasobject or dtype.fields for _, dtype in self._schema):
            raise TypeError("Checkpoints require numeric array leaves")
        self._checkpoints = {}
        self._temporary = None
        self._active = False
        self._ram_bytes = self._disk_bytes = 0
        self._counts = dict(ram=0, disk=0)
        self.statistics = dict(
            ram_saves=0,
            disk_saves=0,
            ram_restores=0,
            disk_restores=0,
            disk_bytes_written=0,
            disk_bytes_read=0,
            peak_ram_checkpoints=0,
            peak_disk_checkpoints=0,
            peak_ram_bytes=0,
            peak_disk_bytes=0,
        )

    def __enter__(self):
        if self._active:
            raise RuntimeError("Checkpoint store is already open")
        self._active = True
        return self

    def __exit__(self, *args):
        self._checkpoints.clear()
        self._ram_bytes = self._disk_bytes = 0
        self._counts = dict(ram=0, disk=0)
        self._active = False
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _require_open(self):
        if not self._active:
            raise RuntimeError("Use CheckpointStore as a context manager")

    def save(self, index, carry, *, storage):
        """Save an independent, lossless copy of every carry leaf."""
        import jax
        import numpy as np

        self._require_open()
        index = _integer(index, "checkpoint index")
        if index in self._checkpoints:
            raise ValueError(f"Checkpoint {index} already exists")
        leaves, structure = jax.tree.flatten(carry)
        schema = [(tuple(leaf.shape), np.dtype(leaf.dtype)) for leaf in leaves]
        if structure != self._structure or schema != self._schema:
            raise ValueError("Checkpoint tree, shapes, or dtypes differ from the template")
        if storage == "ram":
            if self._counts["ram"] >= self.ram_checkpoints:
                raise ValueError("RAM checkpoint budget exceeded")
            if self.max_ram_bytes is not None and self._ram_bytes + self.state_bytes > self.max_ram_bytes:
                raise ValueError("RAM checkpoint byte budget exceeded")
            data = [np.array(jax.device_get(leaf), copy=True) for leaf in leaves]
            size = sum(leaf.nbytes for leaf in data)
            self._ram_bytes += size
        elif storage == "disk":
            if self.directory is None:
                raise ValueError("Disk checkpoints require a directory")
            if self.disk_checkpoints is not None and self._counts["disk"] >= self.disk_checkpoints:
                raise ValueError("Disk checkpoint budget exceeded")
            if self._temporary is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="veros-ad-", dir=self.directory)
            data = Path(self._temporary.name) / f"{index}.npz"
            # Stream leaves so writing does not retain a full host copy on GPU.
            import zipfile

            with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for number, leaf in enumerate(leaves):
                    with archive.open(f"leaf_{number}.npy", "w", force_zip64=True) as output:
                        np.save(output, np.asarray(jax.device_get(leaf)), allow_pickle=False)
            size = data.stat().st_size
            self._disk_bytes += size
            self.statistics["disk_bytes_written"] += size
        else:
            raise ValueError("Checkpoint storage must be ram or disk")
        self._checkpoints[index] = storage, data, size
        self._counts[storage] += 1
        self.statistics[f"{storage}_saves"] += 1
        for tier, size in (("ram", self._ram_bytes), ("disk", self._disk_bytes)):
            count = self._counts[tier]
            self.statistics[f"peak_{tier}_checkpoints"] = max(self.statistics[f"peak_{tier}_checkpoints"], count)
            self.statistics[f"peak_{tier}_bytes"] = max(self.statistics[f"peak_{tier}_bytes"], size)

    def restore(self, index):
        """Materialize a saved carry on the current JAX device."""
        import jax
        import numpy as np

        self._require_open()
        storage, data, size = self._checkpoints[index]
        if storage == "ram":
            leaves = [jax.device_put(leaf) for leaf in data]
        else:
            leaves = []
            with np.load(data, allow_pickle=False) as archive:
                for number, (shape, dtype) in enumerate(self._schema):
                    leaf = archive[f"leaf_{number}"]
                    if leaf.shape != shape or leaf.dtype != dtype:
                        raise ValueError("Disk checkpoint shapes or dtypes differ from the template")
                    leaves.append(jax.device_put(leaf))
            self.statistics["disk_bytes_read"] += size
        self.statistics[f"{storage}_restores"] += 1
        return jax.tree.unflatten(self._structure, leaves)

    def drop(self, index):
        """Release a slot and remove its file when stored on disk."""
        self._require_open()
        storage, data, size = self._checkpoints.pop(index)
        self._counts[storage] -= 1
        if storage == "disk":
            data.unlink()
            self._disk_bytes -= size
        else:
            self._ram_bytes -= size


class CheckpointedIntegrator:
    """First-order reverse integration with host checkpoint replay.

    Jitted steps and one-step VJPs run on device. Host scheduling and I/O prevent
    JAX transformations of ``value_and_grad``. MPI ranks use the same schedule and
    global scalar objectives, with adjoint seeds divided across ranks.
    ``terminal_objective(final)`` returns the final loss. ``record_objective(records)``
    contributes a loss per forward step with a singleton step axis. Both may accept
    ``parameters`` for direct dependence on setting overrides.
    """

    def __init__(self, prepared, *, terminal_objective=None, record_objective=None):
        import jax
        import jax.numpy as jnp
        from veros import runtime_state, runtime_settings

        self._comm = runtime_settings.mpi_comm if runtime_state.proc_num > 1 else None
        self._ranks = runtime_state.proc_num
        if self._comm is not None:
            objectives = self._comm.allgather((terminal_objective is not None, record_objective is not None))
            if any(item != objectives[0] for item in objectives):
                raise ValueError("MPI ranks must agree on objective interfaces")
        if terminal_objective is None and record_objective is None:
            raise ValueError("Provide a terminal_objective or record_objective")
        self.prepared = prepared

        def bind_objective(objective):
            if "parameters" in inspect.signature(objective).parameters:
                return lambda argument, parameters: objective(argument, parameters=parameters)
            return lambda argument, parameters: objective(argument)

        record_loss = None if record_objective is None else bind_objective(record_objective)

        def step(carry, parameters, inputs):
            overrides = {"parameters": parameters} if parameters else {}
            if inputs:
                overrides["step_inputs"] = inputs
            final, records = prepared.step(carry, **overrides)
            if record_loss is not None:
                records = jax.tree.map(lambda value: value[None], records)
            contribution = jnp.asarray(0.0) if record_loss is None else jnp.asarray(record_loss(records, parameters))
            if contribution.ndim:
                raise ValueError("record_objective must return a scalar")
            if not jnp.issubdtype(contribution.dtype, jnp.floating):
                raise ValueError("record_objective must return a floating scalar")
            return final, contribution

        def reverse(carry, parameters, inputs, adjoint):
            (final, contribution), pullback = jax.vjp(lambda c, p: step(c, p, inputs), carry, parameters)
            gradient, parameter_gradient = pullback((adjoint, jnp.ones_like(contribution) / self._ranks))
            return gradient, parameter_gradient, (final.failure_code, final.failed_iteration)

        if terminal_objective is None:

            def terminal_objective(final):
                return jnp.asarray(0.0)

        self._forward = jax.jit(step)
        self._reverse = jax.jit(reverse)
        terminal = bind_objective(terminal_objective)
        self._terminal = jax.jit(
            jax.value_and_grad(lambda c, p: terminal(c, p) / self._ranks, argnums=(0, 1), allow_int=True)
        )

    def _collective(self, function):
        """Propagate host validation and storage failures before the next kernel."""
        if self._comm is None:
            return function()
        result, failure = None, None
        try:
            result = function()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        failures = self._comm.allgather(failure)
        if any(failures):
            raise RuntimeError(f"Distributed checkpoint operation failed: {failures}")
        return result

    @staticmethod
    def _memory_analysis(function, *arguments):
        """Read XLA buffer estimates for a compiled numerical signature."""
        analysis = function.lower(*arguments).compile().memory_analysis()
        if analysis is None:
            return None
        return {
            name: getattr(analysis, f"{field}_size_in_bytes")
            for name, field in (
                ("argument_bytes", "argument"),
                ("output_bytes", "output"),
                ("alias_bytes", "alias"),
                ("temporary_bytes", "temp"),
                ("generated_code_bytes", "generated_code"),
            )
        }

    def value_and_grad(
        self,
        initial,
        *,
        n_steps,
        schedule=None,
        directory=None,
        parameters=None,
        step_inputs=None,
        input_storage="ram",
        max_checkpoint_ram_bytes=None,
        max_input_ram_bytes=None,
        shared_parameters=None,
        shared_variables=(),
        observer=None,
    ):
        """Return loss, carry and setting gradients, final carry, and storage statistics.

        ``gradient`` is the initial-carry cotangent, with ``float0`` leaves for integers
        and Booleans. ``parameter_gradient`` contains setting-override derivatives.
        The prepared integrator's supported inputs and fixed configuration apply.
        ``step_inputs`` is read once and saved in ``input_storage`` for replay.
        ``observer(index, carry)`` runs once per original forward step.
        MPI parameters are shared unless ``shared_parameters=()`` selects rank-local
        parameters. ``shared_variables`` names replicated initial controls whose
        cotangents are summed across ranks. Files are cleaned on success or failure.
        """
        import jax
        import jax.numpy as jnp

        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((initial, parameters))):
            raise RuntimeError(
                "CheckpointedIntegrator.value_and_grad is a host operation and cannot be JAX-transformed"
            )
        parameters = self._collective(
            lambda: {} if parameters is None else self.prepared.validate_parameters(parameters)
        )
        n_steps = self._collective(lambda: _integer(n_steps, "n_steps"))
        schedule = BinomialSchedule(n_steps) if schedule is None else schedule

        def validate():
            if schedule.n_steps != n_steps:
                raise ValueError("Schedule length differs from the number of timesteps")
            if input_storage not in ("ram", "disk"):
                raise ValueError("input_storage must be 'ram' or 'disk'")
            self.prepared.check_status(initial)
            if hasattr(self.prepared, "check_configuration"):
                self.prepared.check_configuration()
            for name in shared_variables:
                if name not in initial.variables:
                    raise ValueError(f"Unknown shared initial variable {name}")
                if not jnp.issubdtype(initial.variables.get(name).dtype, jnp.floating):
                    raise ValueError(f"Shared initial variable {name} must be floating-point")
            return validate_schedule(schedule)

        counts = self._collective(validate)
        shared_parameters = tuple(sorted(parameters if shared_parameters is None else set(shared_parameters)))
        shared_variables = tuple(sorted(set(shared_variables)))
        self._collective(lambda: [parameters[name] for name in shared_parameters])
        if self._comm is not None:
            import hashlib
            import numpy as np

            digest = hashlib.sha256()
            for action in schedule.actions():
                digest.update(repr(action).encode())
            solver_type = type(getattr(getattr(self.prepared, "_template", None), "_linear_solver", None))
            signature = (
                digest.hexdigest(),
                tuple(shared_parameters),
                tuple(shared_variables),
                input_storage,
                step_inputs is not None,
                repr([(name, np.asarray(parameters[name]).tolist()) for name in shared_parameters]),
                repr(getattr(self.prepared, "_configuration", None)),
                repr(getattr(self.prepared, "_runtime_configuration", None)),
                repr(getattr(self.prepared, "_diagnostic_configuration", None)),
                tuple(getattr(self.prepared, "record_diagnostics", ())),
                solver_type.__module__,
                solver_type.__qualname__,
                repr([(leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(initial)]),
                str(jax.tree.structure(parameters)),
                tuple(
                    (name, hashlib.sha256(memoryview(np.ascontiguousarray(initial.variables.get(name)))).hexdigest())
                    for name in shared_variables
                ),
            )
            signatures = self._comm.allgather(signature)
            if any(item != signatures[0] for item in signatures):
                raise ValueError("MPI ranks must agree on configuration, schedules, input storage, and shared controls")
        inputs = None if step_inputs is None else iter(step_inputs)

        def read_inputs():
            try:
                return self.prepared.validate_step_inputs(next(inputs))
            except StopIteration:
                raise ValueError("step_inputs must contain one mapping per timestep") from None

        first_inputs = self._collective(read_inputs) if inputs is not None and n_steps else {}
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(first_inputs)):
            raise RuntimeError(
                "CheckpointedIntegrator.value_and_grad is a host operation and cannot be JAX-transformed"
            )
        if self._comm is not None:
            schemas = self._comm.allgather(
                (
                    str(jax.tree.structure(first_inputs)),
                    repr([(leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(first_inputs)]),
                )
            )
            if any(item != schemas[0] for item in schemas):
                raise ValueError("MPI ranks must agree on forcing input schemas")
        # Trace derivative rules before the original trajectory or any disk writes.
        adjoint_shape = self._collective(lambda: jax.eval_shape(self._terminal, initial, parameters)[1][0])
        self._collective(lambda: jax.eval_shape(self._reverse, initial, parameters, first_inputs, adjoint_shape))
        current, final, adjoint = initial, None, None
        parameter_gradient = jax.tree.map(jnp.zeros_like, parameters)
        value = jnp.asarray(0.0)
        with ExitStack() as stack:
            store = self._collective(
                lambda: stack.enter_context(
                    CheckpointStore(
                        initial,
                        ram_checkpoints=schedule.ram_checkpoints,
                        disk_checkpoints=getattr(schedule, "disk_checkpoints", None),
                        directory=directory,
                        max_ram_bytes=max_checkpoint_ram_bytes,
                    )
                )
            )
            input_store = self._collective(
                lambda: stack.enter_context(
                    CheckpointStore(
                        first_inputs,
                        ram_checkpoints=n_steps,
                        directory=directory,
                        disk_checkpoints=n_steps,
                        max_ram_bytes=max_input_ram_bytes,
                    )
                )
            )
            if self._comm is not None:
                # Serialize compiler peaks across ranks sharing one host.
                for rank in range(self._ranks):

                    def compile_rank():
                        if self._comm.rank == rank:
                            self._forward.lower(initial, parameters, first_inputs).compile()
                            self._reverse.lower(initial, parameters, first_inputs, adjoint_shape).compile()
                            self._terminal.lower(initial, parameters).compile()

                    self._collective(compile_rank)
            for action in schedule.actions():
                if action.kind == "save":
                    self._collective(lambda: store.save(action.index, current, storage=action.storage))
                elif action.kind == "restore":
                    current = self._collective(lambda: store.restore(action.index))
                elif action.kind == "drop":
                    self._collective(lambda: store.drop(action.index))
                elif action.kind == "advance":
                    for index in range(action.index, action.end):
                        if inputs is None:
                            step_input = {}
                        elif final is None:
                            step_input = first_inputs if index == 0 else self._collective(read_inputs)
                            self._collective(lambda: input_store.save(index, step_input, storage=input_storage))
                        else:
                            step_input = self._collective(lambda: input_store.restore(index))
                        current, contribution = self._forward(current, parameters, step_input)
                        self._collective(lambda: self.prepared.check_status(current))
                        if final is None:
                            value = value + contribution
                            if observer is not None:
                                self._collective(lambda: observer(index, current))
                elif action.kind == "turn":
                    if inputs is not None:

                        def check_end():
                            marker = object()
                            if next(inputs, marker) is not marker:
                                raise ValueError("step_inputs contains more mappings than timesteps")

                        self._collective(check_end)
                    final = current
                    terminal_value, (adjoint, parameter_gradient) = self._terminal(final, parameters)
                    value = value + terminal_value * self._ranks
                elif action.kind == "reverse":
                    step_input = {} if inputs is None else self._collective(lambda: input_store.restore(action.index))
                    adjoint, contribution, (code, iteration) = self._reverse(current, parameters, step_input, adjoint)
                    parameter_gradient = jax.tree.map(lambda a, b: a + b, parameter_gradient, contribution)
                    jax.block_until_ready(adjoint)
                    self._collective(
                        lambda: self.prepared.check_status(
                            current._replace(failure_code=code, failed_iteration=iteration)
                        )
                    )
            jax.block_until_ready((value, adjoint, final, parameter_gradient))
            if self._comm is not None:
                import numpy as np
                from mpi4py import MPI

                def reduce_shared(value):
                    source = np.asarray(value)
                    result = np.empty_like(source)
                    self._comm.Allreduce(source, result, op=MPI.SUM)
                    return jnp.asarray(result)

                for name in shared_parameters:
                    parameter_gradient[name] = reduce_shared(parameter_gradient[name])
                if shared_variables:
                    from copy import copy

                    variables = copy(adjoint.variables)
                    replacements = {name: reduce_shared(variables.get(name)) for name in shared_variables}
                    if hasattr(variables, "unlock"):
                        with variables.unlock():
                            variables.update(replacements)
                    else:
                        variables.update(replacements)
                    adjoint = adjoint._replace(variables=variables)
            statistics = dict(
                store.statistics,
                **counts,
                state_bytes=store.state_bytes,
                adjoint_bytes=sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(adjoint)),
                parameter_bytes=sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(parameters)),
                forward_signatures=self._forward._cache_size(),
                reverse_signatures=self._reverse._cache_size(),
                ranks=self._ranks,
                input_peak_ram_bytes=input_store.statistics["peak_ram_bytes"],
                input_peak_disk_bytes=input_store.statistics["peak_disk_bytes"],
                forward_memory=(
                    self._collective(lambda: self._memory_analysis(self._forward, initial, parameters, first_inputs))
                    if n_steps
                    else None
                ),
                reverse_memory=(
                    self._collective(
                        lambda: self._memory_analysis(self._reverse, initial, parameters, first_inputs, adjoint_shape)
                    )
                    if n_steps
                    else None
                ),
            )
        return CheckpointedResult(value, adjoint, final, statistics, parameter_gradient)
