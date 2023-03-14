import pygame
from snake import Snake
from apple import Apple


class Game:
    """Initialize the game"""

    def __init__(self, game_width, game_height):
        pygame.display.set_caption("Snake")
        self.game_width = game_width
        self.game_height = game_height
        self.surface = pygame.display.set_mode(
            (game_width, game_height + 60)
        )  # 60 for score, status bar
        self.crash = False
        self.snake = Snake(self)
        self.apple = Apple()
        self.score = 0

    # If snake eats apple, apple is repositioned
    def eat(self, snake, apple):
        if snake.x == apple.x_pos and snake.y == apple.y_pos:
            apple.get_pos(self, snake)
            snake.eaten = True
            self.score = self.score + 1

    # Get the current score
    def get_record(self, score, record):
        if score >= record:
            return score
        else:
            return record

    # Start the game
    def init_game(self, snake, game, apple, agent, batch_size):
        state_init1 = agent.get_state(game, snake, apple)
        action = [1, 0, 0]
        snake.play_step(action, snake.x, snake.y, game, apple, agent)
        state_init2 = agent.get_state(game, snake, apple)
        reward1 = agent.set_reward(snake, game.crash)
        agent.remember(state_init1, action, reward1, state_init2, game.crash)
        agent.train_long_memory(agent.memory, batch_size)
