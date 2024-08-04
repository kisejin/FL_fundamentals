import argparse
import copy
import logging
import os
import time
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


# Change to the results directory
os.chdir(os.path.join(os.path.dirname(__file__), "PFLlib/results"))


def create_result(list_file):
    """
    Create a nested dictionary of results from HDF5 files.

    :param list_file: List of HDF5 file names
    :return: Nested dictionary with results
    """
    result = {
        alg: {
            ds: {metric: None for metric in ["test_acc", "train_loss"]}
            for ds in ["iid", "niid"]
        }
        for alg in ["fedavg", "fedprox_mu_lt_0", "fedprox_mu_0"]
    }

    for file in list_file:
        file_lower = file.lower()
        with h5py.File(file, "r") as f:
            test_acc, train_loss = f["rs_test_acc"][:], f["rs_train_loss"][:]

        for alg in result:
            if alg in file_lower:
                for ds in ["iid", "niid"]:
                    if ds in file_lower and (
                        ds != "iid" or "niid" not in file_lower
                    ):
                        result[alg][ds]["test_acc"] = test_acc
                        result[alg][ds]["train_loss"] = train_loss

    return result


# Create result dictionaries for different drop rates
dr_0 = create_result(
    [
        "Cifar10_iid_FedAvg_train_c-0_3__drop-0.h5",
        "Cifar10_niid_FedAvg_train_c-0_3__drop-0.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-0__mu-0_001.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-0__mu-0_001.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-0__mu-0.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-0__mu-0.h5",
    ]
)

dr_50 = create_result(
    [
        "Cifar10_iid_FedAvg_train_c-0_3__drop-50.h5",
        "Cifar10_niid_FedAvg_train_c-0_3__drop-50.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-50__mu-0_001.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-50__mu-0_001.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-50__mu-0.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-50__mu-0.h5",
    ]
)

dr_90 = create_result(
    [
        "Cifar10_iid_FedAvg_train_c-0_3__drop-90.h5",
        "Cifar10_niid_FedAvg_train_c-0_3__drop-90.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-90__mu-0_001.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-90__mu-0_001.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-90__mu-0.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-90__mu-0.h5",
    ]
)


def show_point(ax, result):
    """
    Show the first point where the result reaches 0.6 on the plot.

    :param ax: Matplotlib axis object
    :param result: List of result values
    """
    intersection_index = np.argmax(np.array(result) >= 0.6)
    if intersection_index != 0:
        intersection_x = intersection_index + 1
        intersection_y = result[intersection_index]
        ax.plot(intersection_x, intersection_y, "ro", markersize=10)
        ax.annotate(
            f"{intersection_x}",
            xy=(intersection_x, intersection_y),
            xytext=(intersection_x + 0.3, intersection_y + 0.03),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )


def plot_drop_rate(mode="test_acc", output_name="test_acc_plot"):
    """
    Plot results for different drop rates.

    :param mode: 'test_acc' or 'train_loss'
    :param output_name: Name of the output file
    """
    plt.figure(figsize=[40, 30])
    plt.rcParams.update({"font.size": 17})

    titles = ["I.I.D Dataset", "Non-I.I.D Dataset"]
    idx_ds = ["iid", "niid"]
    drop_rates = [0, 50, 90]
    labels = ["FedAvg", r"FedProx ($\mu$=0)", r"FedProx ($\mu$>0)"]
    list_alg = ["fedavg", "fedprox_mu_0", "fedprox_mu_lt_0"]
    colors = ["#ff7f0e", "#e377c2", "#17becf"]
    line_styles = [":", "--", "-"]

    for drop_rate in range(3):
        for idx in range(2):
            ax = plt.subplot(3, 2, 2 * (drop_rate) + idx + 1)

            for alg_idx, alg in enumerate(list_alg):
                result = eval(
                    f"dr_{drop_rates[drop_rate]}['{alg}']['{idx_ds[idx]}']['{mode}']"
                )
                label = labels[alg_idx] if drop_rate == 2 and idx == 1 else None
                ax.plot(
                    np.arange(201),
                    result,
                    line_styles[alg_idx],
                    linewidth=3.0,
                    color=colors[alg_idx],
                    label=label,
                )
                show_point(ax, result)

            if mode == "test_acc":
                ax.axhline(y=0.6, color="gray", linestyle="-.", linewidth=3.0)

            ax.set_xlabel("# Rounds", fontsize=22)
            if idx == 0:
                ax.set_ylabel(
                    (
                        "Testing Accuracy"
                        if mode == "test_acc"
                        else "Training Loss"
                    ),
                    fontsize=22,
                )
            if drop_rate == 0:
                ax.set_title(titles[idx], fontsize=22, fontweight="bold")

            ax.tick_params(color="#dddddd")
            for spine in ax.spines.values():
                spine.set_color("#dddddd")
            ax.set_xlim(-0.1, 201)

    plt.legend(
        frameon=False,
        loc="lower center",
        ncol=3,
        prop=dict(weight="bold"),
        borderaxespad=-0.3,
        fontsize=26,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f"{output_name}.png")


def plot_mu_candidate(output_name):
    """
    Plot results for different mu values.

    :param output_name: Name of the output file
    """
    rounds = [200] * 4
    keys = ["mu_0.001", "mu_0.01", "mu_0.1", "mu_1"]
    labels = [r"$\mu$=0.001", r"$\mu$=0.01", r"$\mu$=0.1", r"$\mu$=1"]

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=[60, 45])
    ax = ax.flatten()

    for i in range(4):
        # Test accuracy
        ax[i * 2].plot(
            np.arange(rounds[i] + 1), results[keys[i]]["test_acc"], "b"
        )
        ax[i * 2].set_xlabel("# Rounds", fontsize=22)
        ax[i * 2].set_ylabel("Test Accuracy", fontsize=22)
        ax[i * 2].set_xlim(-1, rounds[i] + 5)
        ax[i * 2].set_title(labels[i], fontsize=22)
        ax[i * 2].axhline(y=0.6, color="gray", linestyle="-.", linewidth=3.0)

        # Training loss
        ax[i * 2 + 1].plot(
            np.arange(rounds[i] + 1), results[keys[i]]["train_loss"], "b"
        )
        ax[i * 2 + 1].set_xlabel("# Rounds", fontsize=22)
        ax[i * 2 + 1].set_ylabel("Training Loss", fontsize=22)
        ax[i * 2 + 1].set_xlim(-1, rounds[i] + 5)
        ax[i * 2 + 1].set_title(labels[i], fontsize=22)

    plt.tight_layout()
    fig.savefig(f"{output_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type_plot",
        type=str,
        default="plot_drop_rate",
        choices=["plot_mu", "plot_drop_rate"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test_acc",
        choices=["test_acc", "train_loss"],
        help="The mode (accuracy or loss) to plot",
    )
    parser.add_argument("--output_name", type=str, required=True)

    args = parser.parse_args()

    if args.type_plot == "plot_drop_rate":
        plot_drop_rate(args.mode, args.output_name)
    else:
        plot_mu_candidate(args.output_name)
