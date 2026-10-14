# AI-assisted evaluation scaffold: review before submission.

import asyncio
import csv
import random

import numpy as np
import torch

from environment_setup import configure_source
configure_source()

from rooms.room import Room
from agents.random_agent import RandomAgent
from recurrent_agent import RecurrentActorCriticAgent


class EvaluationAgent(RecurrentActorCriticAgent):
    def __init__(self, name, checkpoint_path, **kwargs):
        super().__init__(name=name, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.evaluation_rows = []
        self.last_total = 0.0

    def update_game_start(self, payload):
        super().update_game_start(payload)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["all_actions"] != self.all_actions:
            raise ValueError("Checkpoint action ordering does not match.")

        self.network.load_state_dict(checkpoint["model_state"])
        self.network.eval()
        self.last_total = 0.0

    def request_action(self, payload):
        # Keep stochastic action selection, but disable gradient recording.
        with torch.no_grad():
            return super().request_action(payload)

    def update_match_over(self, payload):
        total = float(payload["scores"][self.name])
        score = total - self.last_total
        self.last_total = total

        if score not in (0.0, 1.0, 2.0, 3.0):
            raise ValueError(f"Unexpected match score: {score}")

        self.evaluation_rows.append({
            "match": len(self.evaluation_rows) + 1,
            "score": score,
            "win": int(score == 3),
        })

        # No loss calculation or optimiser step during evaluation.
        self.trajectory.clear()
        self.network.reset_memory()

    def update_game_over(self, payload):
        with open("recurrent_evaluation.csv", "w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["match", "score", "win"]
            )
            writer.writeheader()
            writer.writerows(self.evaluation_rows)

        scores = [row["score"] for row in self.evaluation_rows]
        wins = [row["win"] for row in self.evaluation_rows]

        print(f"Evaluation matches: {len(scores)}")
        print(f"Mean score: {np.mean(scores):.3f}")
        print(f"Win rate: {np.mean(wins):.1%}")
        print("Saved recurrent_evaluation.csv")


async def main():
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)

    room = Room(
        run_remote_room=False,
        room_name="recurrent_evaluation",
        max_matches=50,
    )

    learner = EvaluationAgent(
        name="Evaluated",
        checkpoint_path="recurrent_checkpoint.pt",
        seed=100,
        log_directory=room.room_dir,
    )
    room.connect_player(learner)

    for i in range(3):
        room.connect_player(
            RandomAgent(
                name=f"Opponent_{i}",
                log_directory=room.room_dir,
            )
        )

    await room.run()


if __name__ == "__main__":
    asyncio.run(main())