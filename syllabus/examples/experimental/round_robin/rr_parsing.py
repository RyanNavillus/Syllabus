import argparse
import json
import os
from json import JSONDecodeError

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import wandb
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        required=True,
        type=str,
        help=(
            "the round robin directory to parse,"
            "should be in the following format:'DR_SP_DR_FSP'"
        ),
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="save the output json and return plots to wandb",
    )
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


def plot_returns(values: pd.Series, agent_name: str):
    df = pd.DataFrame(values)
    df.columns = ["values"]
    df["avg"] = df["values"].apply(lambda x: np.array(x).mean())
    df["std"] = df["values"].apply(lambda x: np.array(x).std())

    df["lower"] = df["avg"] - df["std"]
    df["upper"] = df["avg"] + df["std"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["avg"],
            mode="lines",
            line=dict(color="blue"),
            name="Average",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.array([df.index, df.index[::-1]]).flatten(),
            y=np.array([df["upper"], df["lower"][::-1]]).flatten(),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="Error Band",
        )
    )

    fig.update_layout(
        title=agent_name,
        xaxis_title="Checkpoints",
        yaxis_title="Returns",
        showlegend=True,
    )
    return fig


if __name__ == "__main__":
    args = parse_args()
    directory = f"{args.directory}/round-robin"

    agent_path = directory.split("/")[0].split("_")
    agent_1 = "_".join(agent_path[:2])
    agent_2 = "_".join(agent_path[2:])
    legal_names = ("DR_SP", "DR_FSP", "DR_PFSP", "PLR_SP", "PLR_FSP", "PLR_PFSP")

    for agent in [agent_1, agent_2]:
        assert (
            agent in legal_names
        ), f"Expected agent name in {legal_names} but got {agent}"

    logs = merge_logs(directory, args.logging_dir)

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=f"round_robin__{agent_1}_{agent_2}",
        monitor_gym=True,
        save_code=True,
        dir=args.logging_dir,
    )
    wandb.run.log_code(os.path.join(args.logging_dir))

    if args.save:
        returns_artifact = wandb.Artifact("returns", type="json")
        returns_artifact.add_file(
            os.path.join(args.logging_dir, f"{agent_1}_{agent_2}_returns.json")
        )
        wandb.log_artifact(returns_artifact)

        df = pd.DataFrame(logs).sort_index()
        df[f"{agent_1}"] = df[f"{agent_1}"].apply(lambda x: np.array(x))
        df[f"{agent_2}"] = df[f"{agent_2}"].apply(lambda x: np.array(x))

        fig = plot_returns(df[f"{agent_1}"], agent_1)
        wandb.log({f"charts/{agent_1}_returns": wandb.Html(plotly.io.to_html(fig))})
        fig = plot_returns(df[f"{agent_2}"], agent_2)
        wandb.log({f"charts/{agent_2}_returns": wandb.Html(plotly.io.to_html(fig))})
