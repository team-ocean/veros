import json
import click
import numpy as np

import matplotlib as mpl
import seaborn as sns
mpl.use("agg")

import matplotlib.pyplot as plt  # noqa: E402

sns.set_style("ticks")

COMPONENT_COLORS = {
    "numpy": "orangered",
    "numpy-mpi": "coral",
    "jax": "dodgerblue",
    "jax-mpi": "steelblue",
    "jax-gpu": "teal",
    "fortran": "0.4",
    "fortran-mpi": "0.2",
}


@click.argument("INFILES", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.option("--xaxis", type=click.Choice(["nproc", "size"]), required=True)
@click.option("--norm-component", default=None)
@click.command()
def plot_benchmarks(infiles, xaxis, norm_component):
    benchmarks = set()
    components = set()
    sizes = set()
    nprocs = set()

    for infile in infiles:
        with open(infile) as f:
            data = json.load(f)

        meta = data["settings"]
        benchmarks |= set(meta["only"])
        components |= set(meta["components"])
        sizes |= set(meta["sizes"])
        nprocs.add(meta["nproc"])

    if xaxis == "nproc":
        assert len(sizes) == 1
        xvals = np.array(sorted(nprocs))
    elif xaxis == "size":
        assert len(nprocs) == 1
        xvals = np.array(sorted(sizes))
    else:
        assert False

    if norm_component is not None and norm_component not in components:
        raise ValueError(f"Did not find norm component {norm_component} in data")

    component_data = {
        benchmark: {
            comp: np.full(len(xvals), np.nan)
            for comp in components
        }
        for benchmark in benchmarks
    }

    for infile in infiles:
        with open(infile) as f:
            data = json.load(f)

        for benchmark, bench_res in data["benchmarks"].items():
            for res in bench_res:
                if xaxis == "size":
                    # sizes are approximate, take the closest one
                    x_idx = np.argmin(np.abs(np.array(xvals) - res["size"]))
                else:
                    x_idx = xvals.tolist().index(data["settings"]["nproc"])

                time = float(res["per_iteration"]["mean"])
                component_data[benchmark][res["component"]][x_idx] = time

    for benchmark in benchmarks:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), dpi=150)

        last_coords = {}
        for component in components:
            if norm_component:
                # compute rel. speedup
                yvals = component_data[benchmark][norm_component] / component_data[benchmark][component]
            else:
                yvals = component_data[benchmark][component]

            plt.plot(xvals, yvals, ".--", color=COMPONENT_COLORS[component])

            finite_mask = np.isfinite(yvals)
            if finite_mask.any():
                last_coords[component] = (xvals[finite_mask][-1], yvals[finite_mask][-1])
            else:
                last_coords[component] = (xvals[0], 1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        title_kwargs = dict(
            fontdict=dict(weight="bold", size=11),
            ha="left", x=0, y=1.05
        )
        if xaxis == "nproc":
            plt.xlabel("Number of MPI processes")
            mantissa, exponent = f"{list(sizes)[0]:.1e}".split("e")
            exponent = exponent.lstrip("+0")
            plt.title(f"Benchmark '{benchmark}' for size {mantissa} $\\times$ 10$^{{{exponent}}}$", **title_kwargs)

        elif xaxis == "size":
            nproc = list(nprocs)[0]
            plt.xlabel("Problem size (# elements)")
            plt.title(f"Benchmark '{benchmark}' on {nproc} processes", **title_kwargs)

            if norm_component:
                plt.axhline(nproc, linestyle="dashed", alpha=0.4, lw=1, color="C0")
                plt.annotate("Perfect CPU scaling", (min(xvals), nproc), xytext=(0, 2), textcoords="offset points", alpha=0.4, color="C0")

        if norm_component:
            plt.ylabel("Relative speedup")
        else:
            plt.ylabel("Time per iteration (s)")

        plt.xscale("log")
        plt.yscale("log")

        fig.canvas.draw()

        # add annotations, make sure they don"t overlap
        last_text_pos = 0
        for component, (x, y) in sorted(last_coords.items(), key=lambda k: k[1][1]):
            trans = ax.transData
            _, tp = trans.transform((0, y))
            tp = max(tp, last_text_pos + 15)
            _, y = trans.inverted().transform((0, tp))

            plt.annotate(
                component, (x, y), xytext=(10, 0), textcoords="offset points",
                annotation_clip=False, color=COMPONENT_COLORS[component], va="center",
                weight="bold",
            )

            last_text_pos = tp

        fig.tight_layout()
        fig.savefig(f"{benchmark}.png")
        plt.close(fig)


if __name__ == "__main__":
    plot_benchmarks()
