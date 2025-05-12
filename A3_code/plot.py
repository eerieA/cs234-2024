from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def early_exit(message):
    print(message)
    exit()


def plot_combined(name, results):
    results = np.array(results)
    xs = np.arange(results.shape[1])
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys, label=name)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        default=None,  # "results"
        help="Directory containing the results of the runs",
    )
    parser.add_argument("--dpo-directory", default=None)  # "results_dpo"
    parser.add_argument(
        "-o", "--output", required=True, help="Path in which to save the output image"
    )
    parser.add_argument("--rlhf-directory", default=None)  # "results_rlhf"
    parser.add_argument(
        "--seeds", required=True, help="Comma-separated list of seeds to plot"
    )
    args = parser.parse_args()

    if args.seeds.isdigit():
        seeds = [int(args.seeds)]
    else:
        seeds = [int(seed) for seed in args.seeds.split(",")]

    all_results = {}
    directory = None
    dpo_directory = None
    rlhf_directory = None
    if args.directory:
        directory = Path(args.directory)
        if not directory.is_dir():
            early_exit(f"{directory.resolve()} is not a directory")

        all_results.update({"Early termination": [], "No early termination": []})
    if args.dpo_directory:
        dpo_directory = Path(args.dpo_directory)
        if not dpo_directory.is_dir():
            early_exit(f"{dpo_directory.resolve()} is not a directory")

        all_results.update({"DPO": []})
    if args.rlhf_directory:
        rlhf_directory = Path(args.rlhf_directory)
        if not rlhf_directory.is_dir():
            early_exit(f"{rlhf_directory.resolve()} is not a directory")

        all_results.update({"RLHF (original)": [], "RLHF (learned)": []})
    for seed in seeds:
        if directory is not None:
            format_str = f"Hopper-v3-early-termination={{}}-seed={seed}"
            all_results["Early termination"].append(
                np.load(directory / format_str.format("True") / "scores.npy")
            )
            all_results["No early termination"].append(
                np.load(directory / format_str.format("False") / "scores.npy")
            )
        if dpo_directory is not None:
            all_results["DPO"].append(
                np.load(dpo_directory / f"Hopper-v3-dpo-seed={seed}" / "scores.npy")
            )
        if rlhf_directory is not None:
            all_results["RLHF (original)"].append(
                np.load(
                    rlhf_directory
                    / f"Hopper-v3-rlhf-seed={seed}"
                    / "original_scores.npy"
                )
            )
            all_results["RLHF (learned)"].append(
                np.load(
                    rlhf_directory
                    / f"Hopper-v3-rlhf-seed={seed}"
                    / "learned_scores.npy"
                )
            )

    plt.figure()
    plt.title("Hopper-v3")
    plt.xlabel("Iteration")
    for name, results in all_results.items():
        plot_combined(name, results)
    plt.legend()
    plt.savefig(Path(args.output), bbox_inches="tight")
