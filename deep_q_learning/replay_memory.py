import random
from collections import namedtuple, deque

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "end")
)


class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
