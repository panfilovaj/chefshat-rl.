# AI-assisted implementation, reviewed during development; student review and adaptation required.
import csv
import torch

from torch import nn
from torch.distributions import Categorical
from restricted_agent import RestrictedRandomAgent


class ActorCriticAgent(RestrictedRandomAgent):
    def __init__(self, name, seed=42, learning_rate=0.001, **kwargs):
        super().__init__(name=name, seed=seed, **kwargs)
        torch.manual_seed(seed)

        self.learning_rate = learning_rate
        self.network = None
        self.optimizer = None
        self.trajectory = []
        self.match_count = 0
        self.previous_total_score = 0.0
        self.training_history = []
        self.output_prefix = "feedforward"
    def update_game_start(self, payload):
        super().update_game_start(payload)
        self.previous_total_score = 0.0
        if self.network is None:
            # Input: 12 hand counts plus the legal-action mask.
            input_size = 12 + len(self.all_actions)

            self.network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.Tanh(),
                nn.Linear(64, len(self.all_actions) + 1),
            )
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
            )


    def update_start_match(self, payload):
        super().update_start_match(payload)
        self.trajectory.clear()

    def request_action(self, payload):
        hand_counts, action_mask = self.make_policy_input(payload)

        # Scale counts by the initial hand size of 17.
        hand_tensor = torch.tensor(
            hand_counts / 17.0, dtype=torch.float32
        )
        mask = torch.tensor(action_mask, dtype=torch.bool)

        features = torch.cat([hand_tensor, mask.float()])
        output = self.network(features)

        logits = output[:-1]
        value = output[-1]

        # Illegal actions receive zero probability.
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        distribution = Categorical(logits=masked_logits)
        action = distribution.sample()

        self.trajectory.append(
            (
                distribution.log_prob(action),
                value,
                distribution.entropy(),
            )
        )

        return int(action.item())

    def update_match_over(self, payload):
        total_score = float(payload["scores"][self.name])
        score = total_score - self.previous_total_score
        self.previous_total_score = total_score

        if score not in (0.0, 1.0, 2.0, 3.0):
            raise ValueError(f"Unexpected match score: {score}")

        reward = score / 3.0
        self.match_count += 1

        if not self.trajectory:
            raise RuntimeError("Match ended without agent decisions.")

        log_probs = torch.stack([t[0] for t in self.trajectory])
        values = torch.stack([t[1] for t in self.trajectory])
        entropies = torch.stack([t[2] for t in self.trajectory])

        # Sparse terminal reward, with gamma=1:
        # every decision's return is the final normalised match score.
        returns = torch.full_like(values, reward)
        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (values - returns).square().mean()
        entropy_bonus = entropies.mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        self.training_history.append({
            "match": self.match_count,
            "score": score,
            "reward": reward,
            "decisions": len(self.trajectory),
            "loss": loss.item(),
        })

        print(
            f"Match {self.match_count}: score={score:.0f}, "
            f"decisions={len(self.trajectory)}, "
            f"loss={loss.item():.4f}"
        )
        self.trajectory.clear()

    def update_game_over(self, payload):
        checkpoint_path = f"{self.output_prefix}_checkpoint.pt"
        history_path = f"{self.output_prefix}_history.csv"

        torch.save(
            {
                "model_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "all_actions": self.all_actions,
                "matches_trained": self.match_count,
                "learning_rate": self.learning_rate,
            },
            checkpoint_path,
        )

        with open(history_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["match", "score", "reward", "decisions", "loss"],
            )
            writer.writeheader()
            writer.writerows(self.training_history)

        print(f"Saved {checkpoint_path}")
        print(f"Saved {history_path}")