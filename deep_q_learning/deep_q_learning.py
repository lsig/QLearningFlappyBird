from deep_q_learning.network import DQN
from deep_q_learning.replay_memory import ReplayMemory, Transition
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math


class DeepQLearning:
    _FLAP, _NO_FLAP = 0, 1
    _MIN_V, _MAX_V = -8, 10
    _MAX_X, _MAX_Y = 288, 512
    _ACTIONS = [_FLAP, _NO_FLAP]

    def __init__(
        self,
        n_observations,
        n_actions,
        device,
        batch_size=128,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.01,
        eps_decay=100000,
        tau=0.005,
        lr=1e-4 * 5,
        update_freq=100,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.steps_done = 0
        self.episode_durations = []

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )

        self.memory = ReplayMemory(10000)
        self.N = update_freq
        self.update_counter = 0

    def reward_values(self):
        """returns the reward values used for training

        Note: These are only the rewards used for training.
        The rewards used for evaluating the agent will always be
        1 for passing through each pipe and 0 for all other state
        transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -10.0}

    def observe(self, s1, a, r, s2, end):
        enc_s1 = self._state_encoder(s1)
        enc_s2 = self._state_encoder(s2) if s2 is not None else None
        self._store_transition(enc_s1, a, r, enc_s2, end)
        self._optimize_model()

    def training_policy(self, state):
        return self._select_action(state, training=True)

    def policy(self, state):
        return self._select_action(state, training=False)

    def _store_transition(self, s1, a, r, s2, end):
        # Convert rewards to tensors, and actions to tensors if not already
        s1 = torch.tensor([s1], device=self.device, dtype=torch.float32)
        a = torch.tensor([[a]], device=self.device, dtype=torch.long)
        r = torch.tensor([r], device=self.device, dtype=torch.float32)
        end = torch.tensor([end], device=self.device, dtype=torch.bool)

        # If it's the end of the episode, there is no next state
        s2 = (
            torch.tensor([s2], device=self.device, dtype=torch.float32)
            if not end
            else None
        )

        # Store the transition in replay memory
        self.memory.push(s1, a, r, s2, end)

    def _select_action(self, state, training=False):
        # Private method
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if training and sample < eps_threshold:
            # Exploration: choose a random action
            return random.choice(self._ACTIONS)
        else:
            # Exploitation: choose the best action from policy net
            with torch.no_grad():
                encoded_state = self._state_encoder(state)
                state_tensor = torch.tensor(
                    [encoded_state], device=self.device, dtype=torch.float32
                )
                # Forward pass through the network
                q_values = self.policy_net(state_tensor)
                # Select the action with highest Q-value
                return q_values.max(1)[1].item()

    def _optimize_model(self):
        # Private method
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.update_counter % self.N == 0:
            self._soft_update()
        self.update_counter += 1

    def _soft_update(self, tau=None):
        if tau is None:
            tau = self.tau  # Use the class's tau value if none provided

        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def _state_encoder(self, state):
        y_pos = max(0, min(state["player_y"], 512))
        pipe_top_y = max(25, min(state["next_pipe_top_y"], 192))
        pipe_bottom_y = max(125, min(state["next_pipe_bottom_y"], 292))
        next_pipe_top_y = max(25, min(state["next_next_pipe_top_y"], 192))
        next_pipe_bottom_y = max(125, min(state["next_next_pipe_bottom_y"], 292))
        x_dist = max(0, min(state["next_pipe_dist_to_player"], 288))
        next_x_dist = max(0, min(state["next_next_pipe_dist_to_player"], 288))
        velocity = max(-8, min(state["player_vel"], 10))

        enc_y_pos = (y_pos) / (self._MAX_Y)
        enc_pipe_top_y = pipe_top_y / self._MAX_Y
        enc_pipe_bottom_y = pipe_bottom_y / self._MAX_Y
        enc_x_dist = x_dist / self._MAX_X
        next_enc_pipe_top_y = next_pipe_top_y / self._MAX_Y
        next_enc_pipe_bottom_y = next_pipe_bottom_y / self._MAX_Y
        next_enc_x_dist = next_x_dist / self._MAX_X
        normalized_velocity = (velocity - self._MIN_V) / (self._MAX_V - self._MIN_V)

        return (
            enc_y_pos,
            enc_pipe_top_y,
            enc_pipe_bottom_y,
            enc_x_dist,
            next_enc_pipe_top_y,
            next_enc_pipe_bottom_y,
            next_enc_x_dist,
            normalized_velocity,
        )

    def save_model(self, file_name="target_model.pth"):
        torch.save(self.target_net.state_dict(), file_name)
        print(f"Model saved to {file_name}.")

    def load_model(self, file_name="target_model.pth"):
        self.target_net.load_state_dict(torch.load(file_name))
        self.target_net.eval()
        print(f"Model loaded from {file_name}.")
