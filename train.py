# AI-assisted experiment runner: review and adapt before submission.

import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np
import torch

from environment_setup import configure_source
configure_source()

from rooms.room import Room
from agents.random_agent import RandomAgent
from learning_agent import ActorCriticAgent
from recurrent_agent import RecurrentActorCriticAgent


async def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    run_name = f"{args.agent}_seed{args.seed}_matches{args.matches}"
    prefix = output_dir / run_name

    # Prevent accidentally overwriting an existing experiment.
    if Path(f"{prefix}_checkpoint.pt").exists():
        raise FileExistsError(f"Experiment already exists: {run_name}")

    room = Room(
        run_remote_room=False,
        room_name=run_name,
        max_matches=args.matches,
    )

    agent_class = (
        ActorCriticAgent
        if args.agent == "feedforward"
        else RecurrentActorCriticAgent
    )

    learner = agent_class(
        name="Learner",
        seed=args.seed,
        learning_rate=args.lr,
        log_directory=room.room_dir,
    )
    learner.output_prefix = str(prefix)

    players = [
        learner,
        *[
            RandomAgent(
                name=f"Opponent_{i}",
                log_directory=room.room_dir,
            )
            for i in range(3)
        ],
    ]

    for player in players:
        # Suppress detailed event logs; retain training summaries.
        player.log = lambda message: None
        room.connect_player(player)

    with open(f"{prefix}_config.json", "w") as file:
        json.dump(vars(args), file, indent=2)

    print(f"Starting {run_name}")
    await room.run()
    print("Final scores:", room.final_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["feedforward", "recurrent"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matches", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    asyncio.run(train(parser.parse_args()))