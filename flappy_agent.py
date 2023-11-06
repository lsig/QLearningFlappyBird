# from q_learning.q_learning_optimized import QLearning
from q_learning.q_learning import QLearning
import matplotlib.pyplot as plt
from deep_q_learning.deep_q_learning import DeepQLearning
from ple.games.flappybird import FlappyBird
from ple import PLE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class FlappyAgent:
    def __init__(self, learner):
        self.learner = learner

    def reward_values(self):
        """returns the reward values used for training

        Note: These are only the rewards used for training.
        The rewards used for evaluating the agent will always be
        1 for passing through each pipe and 0 for all other state
        transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        """this function is called during training on each step of the game where
        the state transition is going from state s1 with action a to state s2 and
        yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

        Unless a terminal state was reached, two subsequent calls to observe will be for
        subsequent steps in the same episode. That is, s1 in the second call will be s2
        from the first call.
        """
        return self.learner.observe(s1, a, r, s2, end)

    def training_policy(self, state):
        """Returns the index of the action that should be done in state while training the agent.
        Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

        training_policy is called once per frame in the game while training
        """
        return self.learner.training_policy(state)

    def policy(self, state):
        """Returns the index of the action that should be done in state when training is completed.
        Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

        policy is called once per frame in the game (30 times per second in real-time)
        and needs to be sufficiently fast to not slow down the game.
        """
        return self.learner.policy(state)

    def save_model(self):
        return self.learner.save_model()

    def load_model(self, file_name="target_model.pth"):
        return self.learner.load_model(file_name)


def check_ninety_percent_over_fifty(array):
    total_elements = len(array)
    count_over_fifty = sum(1 for element in array if element >= 50)
    percentage_over_fifty = (count_over_fifty / total_elements) * 100
    return percentage_over_fifty >= 90


def run_game(nb_episodes, agent):
    """Runs nb_episodes episodes of the game with agent picking the moves.
    An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {
        "positive": 1.0,
        "negative": 0.0,
        "tick": 0.0,
        "loss": 0.0,
        "win": 0.0,
    }
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values

    env = PLE(
        FlappyBird(),
        fps=30,
        display_screen=False,
        force_fps=True,
        rng=None,
        reward_values=reward_values,
    )
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True
    env.init()
    scores = []
    total = nb_episodes

    max_score = float("-inf")
    score = 0
    with tqdm(total=nb_episodes, position=0, leave=True) as pbar:
        while nb_episodes > 0:
            # pick an action
            # TODO: for training using agent.training_policy instead
            action = agent.policy(env.game.getGameState())

            # step the environment
            reward = env.act(env.getActionSet()[action])
            # print("reward=%d" % reward)

            # TODO: for training let the agent observe the current state transition

            score += reward

            # reset the environment if the game is over
            if env.game_over():
                if score > 1000 and score > max_score:
                    agent.save_model()
                    max_score = score

                if nb_episodes % 20 == 0 and scores != []:
                    print(
                        f"Max score for {total - nb_episodes} episodes: {max(scores)}"
                    )
                    print(
                        f"Avg score for {total - nb_episodes} episodes: {sum(scores) / len(scores)}"
                    )

                pbar.update(1)
                scores.append(score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
    print(f"90% over 50: {check_ninety_percent_over_fifty(scores)}")
    print(f"Max Test score: {max(scores)}")


def train(nb_episodes, agent):
    reward_values = agent.reward_values()

    env = PLE(
        FlappyBird(),
        fps=30,
        display_screen=False,
        force_fps=True,
        rng=None,
        reward_values=reward_values,
    )
    env.init()

    frames = 0
    total_rewards = []
    scores = []
    total = nb_episodes

    score = 0
    with tqdm(total=nb_episodes, position=0, leave=True) as pbar:
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = agent.training_policy(state)

            # step the environment
            reward = env.act(env.getActionSet()[action])
            # print("reward=%d" % reward)

            # let the agent observe the current state transition
            newState = env.game.getGameState()
            agent.observe(state, action, reward, newState, env.game_over())
            frames += 1

            score += reward

            # reset the environment if the game is over
            if env.game_over():
                if nb_episodes % 1000 == 0 and scores != []:
                    print(
                        f"Max score for {total - nb_episodes} episodes: {max(scores)}"
                    )

                    print(
                        f"Avg score for {total - nb_episodes} episodes: {sum(scores) / len(scores)}"
                    )
                    scores = []
                if score > 1000:
                    break
                scores.append(score)
                total_rewards.append(score)
                env.reset_game()
                pbar.update(1)
                nb_episodes -= 1
                score = 0
        print(f"Max Train score: {max(scores)}")
        print(f"Avg score for {total} episodes: {sum(scores) / len(scores)}")

    plt.plot(total_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    print(f"Number of Frames: {frames}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = FlappyAgent(learner=QLearning())
train(20000, agent)
# agent.load_model("./models/q_table18000.json")
run_game(100, agent)
