import torch.nn.functional as F
import torch.optim as optim
import collections
import numpy as np
import random
import torch

from operator import add
from models import Linear_QNet, Conv_QNet, Deep_QNet


class Agent:
    def __init__(self, options):
        self.model = self.model_selection(options)
        self.model.load_state_dict(torch.load(options.weights_path))
        self.memory = collections.deque(maxlen=options.memory_size)
        self.reward = 0
        self.gamma = 0.9
        self.learning_rate = options.learning_rate
        self.epsilon = 1
        self.optimizer = optim.Adam(
            self.model.parameters(), weight_decay=0, lr=options.learning_rate
        )

    def model_selection(self, options, model="linear"):
        """
        Select the model to use.
        """
        if model == "linear":
            return Linear_QNet(options)
        elif model == "conv":
            return Conv_QNet()
        elif model == "deep":
            return Deep_QNet(options)
        else:
            raise Exception("Model not found.")

    def get_state(self, game, snake, apple):
        """
        Return the state.
        The state is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the right
            - Danger 1 OR 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side
        """
        state = [
            (
                snake.x_change == 20
                and snake.y_change == 0
                and (
                    (
                        list(map(add, snake.position[-1], [snake.segment_width, 0]))
                        in snake.position
                    )
                    or snake.position[-1][0] + snake.segment_width
                    >= (game.game_width - snake.segment_width)
                )
            )
            or (
                snake.x_change == -snake.segment_width
                and snake.y_change == 0
                and (
                    (
                        list(map(add, snake.position[-1], [-snake.segment_width, 0]))
                        in snake.position
                    )
                    or snake.position[-1][0] - snake.segment_width < snake.segment_width
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == -snake.segment_height
                and (
                    (
                        list(map(add, snake.position[-1], [0, -snake.segment_height]))
                        in snake.position
                    )
                    or snake.position[-1][-1] - snake.segment_width
                    < snake.segment_width
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == snake.segment_height
                and (
                    (
                        list(map(add, snake.position[-1], [0, snake.segment_height]))
                        in snake.position
                    )
                    or snake.position[-1][-1] + snake.segment_width
                    >= (game.game_height - snake.segment_height)
                )
            ),  # danger straight
            (
                snake.x_change == 0
                and snake.y_change == -snake.segment_height
                and (
                    (
                        list(map(add, snake.position[-1], [snake.segment_width, 0]))
                        in snake.position
                    )
                    or snake.position[-1][0] + snake.segment_width
                    > (game.game_width - snake.segment_width)
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == snake.segment_height
                and (
                    (
                        list(map(add, snake.position[-1], [-snake.segment_width, 0]))
                        in snake.position
                    )
                    or snake.position[-1][0] - snake.segment_height
                    < snake.segment_height
                )
            )
            or (
                snake.x_change == -snake.segment_width
                and snake.y_change == 0
                and (
                    (
                        list(map(add, snake.position[-1], [0, -snake.segment_height]))
                        in snake.position
                    )
                    or snake.position[-1][-1] - snake.segment_height
                    < snake.segment_height
                )
            )
            or (
                snake.x_change == snake.segment_width
                and snake.y_change == 0
                and (
                    (
                        list(map(add, snake.position[-1], [0, snake.segment_height]))
                        in snake.position
                    )
                    or snake.position[-1][-1] + snake.segment_height
                    >= (game.game_height - snake.segment_height)
                )
            ),  # danger right
            (
                snake.x_change == 0
                and snake.y_change == snake.segment_height
                and (
                    (
                        list(map(add, snake.position[-1], [snake.segment_width, 0]))
                        in snake.position
                    )
                    or snake.position[-1][0] + snake.segment_width
                    > (game.game_width - snake.segment_width)
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == -snake.segment_height
                and (
                    (
                        list(map(add, snake.position[-1], [-snake.segment_width, 0]))
                        in snake.position
                    )
                    or snake.position[-1][0] - snake.segment_width < snake.segment_width
                )
            )
            or (
                snake.x_change == snake.segment_width
                and snake.y_change == 0
                and (
                    (
                        list(map(add, snake.position[-1], [0, -snake.segment_height]))
                        in snake.position
                    )
                    or snake.position[-1][-1] - snake.segment_height
                    < snake.segment_height
                )
            )
            or (
                snake.x_change == -snake.segment_width
                and snake.y_change == 0
                and (
                    (
                        list(map(add, snake.position[-1], [0, snake.segment_height]))
                        in snake.position
                    )
                    or snake.position[-1][-1] + snake.segment_height
                    >= (game.game_height - snake.segment_height)
                )
            ),  # danger left
            snake.x_change == -20,  # move left
            snake.x_change == 20,  # move right
            snake.y_change == -20,  # move up
            snake.y_change == 20,  # move down
            apple.x_pos < snake.x,  # food left
            apple.x_pos > snake.x,  # food right
            apple.y_pos < snake.y,  # food up
            apple.y_pos > snake.y,  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def prediction(self, state):
        """
        Return the prediction of the model.
        The prediction is a numpy array of 4 values, representing:
            - The probability of moving left
            - The probability of moving right
            - The probability of moving up
            - The probability of moving down
        """
        return self.model(state)

    def set_reward(self, snake, crash):
        """
        Return the reward.
        The reward is:
            -10 when Snake crashes.
            +10 when Snake eats food
            0 otherwise
        """
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if snake.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, memory, batch_size):
        """
        Train the model using the replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.model.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(
                np.expand_dims(next_state, 0), dtype=torch.float32
            )
            state_tensor = torch.tensor(
                np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True
            )
            if not done:
                target = reward + self.gamma * torch.max(
                    self.model.forward(next_state_tensor)[0]
                )
            output = self.model.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the model agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.model.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(
            next_state.reshape((1, 11)), dtype=torch.float32
        )
        state_tensor = torch.tensor(
            state.reshape((1, 11)), dtype=torch.float32, requires_grad=True
        )
        if not done:
            target = reward + self.gamma * torch.max(
                self.model.forward(next_state_tensor[0])
            )
        output = self.model.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()
