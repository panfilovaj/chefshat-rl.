# AI-assisted baseline evaluation: review before submission.

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


class BaselineAgent(RandomAgent):
    def __init__(self, name, uniform=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.uniform = uniform
        self.last_total = 0.0
        self.rows = []

    def request_action(self, payload):
        if not self.uniform:
            return super().request_action(payload)

        chosen = random.choice(payload["possible_actions"])
        return self.all_actions.index(chosen)

    def update_match_over(self, payload):
        total = float(payload["scores"][self.name])
        score = total - self.last_total
        self.last_total = total

        if score not in (0.0, 1.0, 2.0, 3.0):
            raise ValueError(f"Unexpected match score: {score}")

        self.rows.append({
            "match": len(self.rows) + 1,
            "score": score,
            "win": int(score == 3),
        })


async def main():
    output_dir = Path("results") / "evaluation"
    output_dir.mkdir(exist_ok=True)
    summary = []

    for baseline in ["provided_random", "uniform_random"]:
        for seed in [100, 101, 102]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            run_name = f"{baseline}_eval{seed}"
            room = Room(
                run_remote_room=False,
                room_name=run_name,
                max_matches=100,
            )

            tested = BaselineAgent(
                name="Evaluated",
                uniform=(baseline == "uniform_random"),
                log_directory=room.room_dir,
            )

            players = [
                tested,
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

            if len(tested.rows) != 100:
                raise RuntimeError("Unexpected match count.")

            with open(output_dir / f"{run_name}.csv", "w", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["match", "score", "win"]
                )
                writer.writeheader()
                writer.writerows(tested.rows)

            mean_score = float(np.mean([r["score"] for r in tested.rows]))
            win_rate = float(np.mean([r["win"] for r in tested.rows]))

            summary.append({
                "baseline": baseline,
                "evaluation_seed": seed,
                "matches": len(tested.rows),
                "mean_score": mean_score,
                "win_rate": win_rate,
            })

            print(
                f"{run_name}: mean score={mean_score:.3f}, "
                f"win rate={win_rate:.1%}"
            )

            with open(
                output_dir / "baseline_summary.csv", "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(summary[0])
                )
                writer.writeheader()
                writer.writerows(summary)

    print("Saved results/evaluation/baseline_summary.csv")


if __name__ == "__main__":
    asyncio.run(main())