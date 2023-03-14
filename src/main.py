import os
import time
import pygame
import numpy as np
from random import randint
import random

import torch

from game import Game
from render import Render
from config import Options
from agent import Agent
from menu import Menu


class Run:
    def __init__(self, options):
        self.render = Render()
        self.agent = Agent(options)
        self.agent = self.agent
        self.model = self.agent.model
        self.counter_games = 0
        self.record = 0
        self.total_score = 0
        self.score_plot = []
        self.counter_plot = []
        self.start_clock = time.time()

    def play(self, training):
        pygame.init()

        while self.counter_games < options.num_games:
            game = Game(options.game_width, options.game_height)
            snake = game.snake
            apple = game.apple

            # Perform first move
            game.init_game(snake, game, apple, self.agent, options.batch_size)

            steps = 0  # steps in the game
            while (not game.crash) and (steps < 100):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            quit()

                if training:
                    # agent.epsilon is set to decay by each step, create less random moves over time
                    self.agent.epsilon = 1 - (self.counter_games * 0.01)
                else:
                    self.agent.epsilon = 0.01

                # get old state
                state_old = self.agent.get_state(game, snake, apple)

                # Random action or action from neural network
                if random.uniform(0, 1) < self.agent.epsilon:
                    final_move = np.eye(3)[randint(0, 2)]
                else:
                    # Predict action based on the old state
                    with torch.no_grad():
                        state_old_tensor = torch.tensor(
                            state_old.reshape((1, 11)), dtype=torch.float32
                        )
                        prediction = self.agent.prediction(state_old_tensor)
                        final_move = np.eye(3)[
                            np.argmax(prediction.detach().cpu().numpy()[0])
                        ]

                # Perform new move and get new state
                snake.play_step(
                    final_move,
                    snake.x,
                    snake.y,
                    game,
                    apple,
                    self.agent,
                )
                # Set reward for the new state
                reward = self.agent.set_reward(snake, game.crash)

                # If food is eaten, steps is set to 0
                if reward > 0:
                    steps = 0

                if training:
                    state_new = self.agent.get_state(game, snake, apple)
                    self.agent.train_short_memory(
                        state_old, final_move, reward, state_new, game.crash
                    )
                    # Store the new data into a long term memory
                    self.agent.remember(
                        state_old, final_move, reward, state_new, game.crash
                    )
                else:
                    record = game.get_record(game.score, self.record)
                    self.render.display(snake, apple, game, record)
                    pygame.time.wait(options.delay)

                steps += 1

            self.counter_games += 1
            self.total_score += game.score
            print(f"Game {self.counter_games}      Score: {game.score}")

            if training:
                self.agent.train_long_memory(self.agent.memory, options.batch_size)
                self.score_plot.append(game.score)
                self.counter_plot.append(self.counter_games)

        if training:
            stop_clock = time.time()
            model_weights = self.model.state_dict()
            if not os.path.exists(options.weights_path):
                os.makedirs(options.weights_path)
            torch.save(model_weights, options.weights_path)
            options.plot(
                self.counter_plot, self.score_plot, self.start_clock, stop_clock
            )
        return self.total_score


if __name__ == "__main__":
    pygame.font.init()
    options = Options()
    run = Run(options)
    agent = Agent(options)

    agent.model = agent.model_selection(options)

    menu = Menu(options, options.game_width, options.game_height + 60)
    init = menu.intro(run.play, agent.model)
