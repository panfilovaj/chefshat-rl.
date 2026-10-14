# AI-assisted implementation, reviewed during development; student review and adaptation required.
import random
import numpy as np

from agents.random_agent import RandomAgent


class RestrictedRandomAgent(RandomAgent):
    def __init__(self, name, seed=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.rng = random.Random(seed)
        self.printed_observation = False

    def make_policy_input(self, payload):
        """Expose own-hand counts and legality; exclude the board."""
        hand = np.asarray(payload["hand"], dtype=np.int64)

        # Values 1–12 represent cards. Zero is padding.
        if np.any((hand < 0) | (hand > 12)):
            raise ValueError("Unexpected card value: check the encoding.")

        hand_counts = np.array(
            [np.count_nonzero(hand == rank) for rank in range(1, 13)],
            dtype=np.float32,
        )

        legal_actions = set(payload["possible_actions"])
        unknown_actions = legal_actions.difference(self.all_actions)
        if unknown_actions:
            raise ValueError(f"Unknown actions: {unknown_actions}")

        action_mask = np.array(
            [action in legal_actions for action in self.all_actions],
            dtype=bool,
        )

        if not action_mask.any():
            raise ValueError("No legal actions received.")

        return hand_counts, action_mask

    def choose_action(self, hand_counts, action_mask):
        """Temporary random policy using only restricted inputs."""
        legal_indices = np.flatnonzero(action_mask).tolist()
        non_pass_indices = [
            i for i in legal_indices if self.all_actions[i] != "pass"
        ]
        return self.rng.choice(non_pass_indices or legal_indices)

    def request_action(self, payload):
        hand_counts, action_mask = self.make_policy_input(payload)

        if not self.printed_observation:
            print(f"{self.name}: hand counts = {hand_counts.tolist()}")
            print(f"Action space size = {len(action_mask)}")
            print(f"Legal actions this turn = {int(action_mask.sum())}")
            print("Board excluded from policy input.")
            self.printed_observation = True

        action_index = self.choose_action(hand_counts, action_mask)

        if not action_mask[action_index]:
            raise ValueError("Policy selected an illegal action.")

        return int(action_index)