# AI-assisted implementation, reviewed during development; student review and adaptation required.
import torch
from torch import nn

from learning_agent import ActorCriticAgent


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size, action_count):
        super().__init__()
        self.encoder = nn.Linear(input_size, 64)
        self.memory = nn.GRUCell(64, 64)
        self.output = nn.Linear(64, action_count + 1)
        self.hidden = None

    def reset_memory(self):
        self.hidden = None

    def forward(self, features):
        encoded = torch.tanh(self.encoder(features)).unsqueeze(0)

        if self.hidden is None:
            self.hidden = encoded.new_zeros((1, 64))

        # Keep the graph across decisions for training through time.
        self.hidden = self.memory(encoded, self.hidden)
        return self.output(self.hidden).squeeze(0)


class RecurrentActorCriticAgent(ActorCriticAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_prefix = "recurrent"

    def update_game_start(self, payload):
        super().update_game_start(payload)

        if not isinstance(self.network, RecurrentNetwork):
            self.network = RecurrentNetwork(
                input_size=12 + len(self.all_actions),
                action_count=len(self.all_actions),
            )
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
            )

        self.network.reset_memory()

    def update_start_match(self, payload):
        super().update_start_match(payload)
        self.network.reset_memory()

    def update_match_over(self, payload):
        # The inherited method trains using the completed trajectory.
        super().update_match_over(payload)
        self.network.reset_memory()