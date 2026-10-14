# AI-assisted experiment runner: review and adapt before submission.

import asyncio
import csv
import random
from pathlib import Path

import numpy as np
import torch

from environment_setup import configure_source
configure_source()

from rooms.room import Room
from agents.random_agent import RandomAgent
from evaluate_feedforward import EvaluationAgent as FeedforwardEvaluator
from evaluate import EvaluationAgent as RecurrentEvaluator


class SaveEvaluation:
    def update_game_start(self, payload):
        super().update_game_start(payload)
        self.original_weights = {
            name: value.detach().clone()
            for name, value in self.network.state_dict().items()
        }

    def update_game_over(self, payload):
        # Verify that evaluation has not changed model weights.
        for name, value in self.network.state_dict().items():
            if not torch.equal(value, self.original_weights[name]):
                raise RuntimeError("Model weights changed during evaluation.")

        with open(self.result_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["match", "score", "win"]
            )
            writer.writeheader()
            writer.writerows(self.evaluation_rows)


class FrozenFeedforward(SaveEvaluation, FeedforwardEvaluator):
    pass


class FrozenRecurrent(SaveEvaluation, RecurrentEvaluator):
    pass


async def main():
    torch.set_num_threads(1)
    output_dir = Path("results") / "evaluation"
    output_dir.mkdir(exist_ok=True)
    summary = []

    # Check all models exist before starting.
    for agent_type in ["feedforward", "recurrent"]:
        for training_seed in [0, 1, 2]:
            checkpoint = Path("results") / (
                f"{agent_type}_seed{training_seed}_matches500_checkpoint.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

    for agent_type in ["feedforward", "recurrent"]:
        evaluator = (
            FrozenFeedforward
            if agent_type == "feedforward"
            else FrozenRecurrent
        )

        for training_seed in [0, 1, 2]:
            checkpoint = Path("results") / (
                f"{agent_type}_seed{training_seed}_matches500_checkpoint.pt"
            )

            for evaluation_seed in [100, 101, 102]:
                random.seed(evaluation_seed)
                np.random.seed(evaluation_seed)
                torch.manual_seed(evaluation_seed)

                run_name = (
                    f"{agent_type}_train{training_seed}"
                    f"_eval{evaluation_seed}"
                )

                room = Room(
                    run_remote_room=False,
                    room_name=run_name,
                    max_matches=100,
                )

                learner = evaluator(
                    name="Evaluated",
                    checkpoint_path=str(checkpoint),
                    seed=evaluation_seed,
                    log_directory=room.room_dir,
                )
                learner.result_path = output_dir / f"{run_name}.csv"

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
                    player.log = lambda message: None
                    room.connect_player(player)

                await room.run()

                rows = learner.evaluation_rows
                if len(rows) != 100:
                    raise RuntimeError("Unexpected evaluation match count.")

                mean_score = float(np.mean([r["score"] for r in rows]))
                win_rate = float(np.mean([r["win"] for r in rows]))

                summary.append({
                    "agent": agent_type,
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "matches": len(rows),
                    "mean_score": mean_score,
                    "win_rate": win_rate,
                })

                print(
                    f"{run_name}: mean score={mean_score:.3f}, "
                    f"win rate={win_rate:.1%}"
                )

                # Save progress after every completed evaluation.
                with open(output_dir / "summary.csv", "w", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=list(summary[0])
                    )
                    writer.writeheader()
                    writer.writerows(summary)

    print("Finished. Saved results/evaluation/summary.csv")


if __name__ == "__main__":
    asyncio.run(main())