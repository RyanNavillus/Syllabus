import argparse
import json
import os
from json import JSONDecodeError

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=".",
        help="the base directory for logging and wandb storage.",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="syllabus",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="rpegoud",
        help="the entity (team) of wandb's project",
    )

    args = parser.parse_args()
    return args


def correct_json_format(path: str, overwrite: bool = False):
    """Converts a round-robin output to correct json format."""
    try:
        with open(path) as f:
            logs = json.load(f)
        return logs

    except JSONDecodeError:
        agent_path = path.split("/")[0].split("_")
        agent_1 = "_".join(agent_path[:2])
        agent_2 = "_".join(agent_path[2:])

        ckp_path = path.split(".")[0].split("/")[-1].split("_")
        ckp_1 = ckp_path[0]
        ckp_2 = ckp_path[2]

        with open(path, "r") as file:
            lines = file.readlines()
        data = [eval(line.strip()) for line in lines]
        assert len(data) == 2, "More than two entries"
        new_data = {
            agent_1: {ckp_1: data[0]},
            agent_2: {ckp_2: data[1]},
        }

        if overwrite:
            with open(path, "w") as output_file:
                json.dump(new_data, output_file, indent=4)

        return new_data


def merge_logs(dir: str, log_dir: str):
    """
    Parses the matchup folder and returns a
    dictionary of returns for each agent/checkpoint pair
    """
    agent_path = dir.split("/")[0].split("_")
    agent_1 = "_".join(agent_path[:2])
    agent_2 = "_".join(agent_path[2:])

    merged_logs = {agent_1: {}, agent_2: {}}

    for file in tqdm(os.listdir(dir)):
        path = os.path.join(dir, file).replace(chr(92), "/")
        path = path.replace("\\", "/")
        try:
            with open(os.path.join(dir, file)) as f:
                logs = json.load(f)
        except JSONDecodeError:
            correct_json_format(path, overwrite=True)
            with open(os.path.join(dir, file)) as f:
                logs = json.load(f)
        for agent in [agent_1, agent_2]:
            for checkpoint, returns in logs[agent].items():
                if checkpoint not in merged_logs[agent].keys():
                    merged_logs[agent][int(checkpoint)] = []
                merged_logs[agent][int(checkpoint)].extend([x for x in returns])

    with open(
        os.path.join(log_dir, f"{agent_1}_{agent_2}_returns.json"),
        "w",
    ) as output_file:
        json.dump(merged_logs, output_file, indent=4)

    return merged_logs


def plot_results(data: dict):

    LABELSIZE = 36
    TICKLABELSIZE = 24

    fig, ax = plt.subplots(figsize=(7 * 2, 3.4 * 2))
    color_palette = "colorblind"
    color_palette = sns.color_palette("deep", n_colors=len(data.keys()))
    colors = dict(zip(data.keys(), color_palette))

    for alg in data.keys():
        df = pd.DataFrame(data[alg])
        df.columns = [int(col) for col in df.columns]
        df = df.T.sort_index()
        avg = df.mean(axis=1)
        std = df.std(axis=1)
        plt.plot(
            df.index,
            avg,
            color=colors[alg],
            marker="o",
            linewidth=2,
            label=alg,
        )
        plt.fill_between(
            df.index,
            y1=avg + std,
            y2=avg - std,
            color=colors[alg],
            alpha=0.2,
        )

    plt.legend(
        loc="lower center",
        ncol=len(data.keys()),
        bbox_to_anchor=(0.5, 1.0),
        fontsize=30,
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    ax.tick_params(length=0.1, width=0.1, labelsize=TICKLABELSIZE)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlabel("Steps", fontsize=LABELSIZE)
    ax.set_ylabel("Mean Episodic Return", fontsize=LABELSIZE)
    fig.tight_layout()
    ax.grid(True, alpha=0.2)

    return fig


def sort_and_trim_dicts(data: dict) -> dict:
    min_length = min(len(sub_dict) for sub_dict in data.values())

    def sort_and_trim(sub_dict, N):
        sorted_keys = sorted(sub_dict.keys(), key=lambda x: int(x))[:N]
        return {k: sub_dict[k] for k in sorted_keys}

    trimmed_data = {
        key: sort_and_trim(sub_dict, min_length) for key, sub_dict in data.items()
    }

    return trimmed_data


def check_dir_name(dir: str, legal_names: list) -> bool:
    split = dir.split("_")
    return all([x in legal_names for x in split])


if __name__ == "__main__":
    args = parse_args()
    legal_names = ("DR", "PLR", "SP", "FSP", "PFSP")
    legal_combinations = ("DR_SP", "DR_FSP", "DR_PFSP", "PLR_SP", "PLR_FSP", "PLR_PFSP")
    valid_dirs = [dir for dir in os.listdir() if check_dir_name(dir, legal_names)]
    print(f"valid directories: {valid_dirs}")

    os.makedirs(args.logging_dir, exist_ok=True)

    # parse the round-robin results, convert to json and save to logging dir
    print("formatting round-robin results ...")
    for dir in valid_dirs:
        print(dir)
        agent_path = dir.split("_")
        agent_1 = "_".join(agent_path[:2])
        agent_2 = "_".join(agent_path[2:])

        for agent in [agent_1, agent_2]:
            assert (
                agent in legal_combinations
            ), f"Expected agent name in {legal_combinations} but got {agent}"

        logs = merge_logs(dir, args.logging_dir)

    # merge all the logs in logging dir and plot the result
    print("merging results ...")
    df = {}
    logs = [file for file in os.listdir(args.logging_dir) if ".json" in file]
    for log in logs:
        print(log)
        with open(os.path.join(args.logging_dir, log), "r") as f:
            returns = json.load(f)

            agent_1, agent_2 = list(returns.keys())
            if agent_1 not in df.keys():
                df[agent_1] = returns[agent_1]
            if agent_2 not in df.keys():
                df[agent_2] = returns[agent_2]
            else:
                for checkpoint in list(returns[agent_1].keys()):
                    df[agent_1][checkpoint].extend(returns[agent_1][checkpoint])
                for checkpoint in list(returns[agent_2].keys()):
                    df[agent_2][checkpoint].extend(returns[agent_2][checkpoint])

    with open(
        os.path.join(args.logging_dir, "merged_rr_returns.json"),
        "w",
    ) as output_file:
        json.dump(df, output_file, indent=4)

    fig = plot_results(df)
    fig.savefig(f"{args.logging_dir}/rr_returns.png", bbox_inches="tight")

    print("logging ...")
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name="round_robin_results",
        save_code=True,
        dir=args.logging_dir,
    )

    wandb.run.log_code(os.path.join(args.logging_dir))
    returns_artifact = wandb.Artifact("returns", type="json")
    returns_artifact.add_file(os.path.join(args.logging_dir, "merged_rr_returns.json"))
    wandb.log_artifact(returns_artifact)
    wandb.log({"charts/rr_returns": wandb.Image(fig)})
