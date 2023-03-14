import pygame
import numpy as np
from render import Render


class Snake(object):
    def __init__(
        self,
        game,
    ):
        self.render = Render()
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.segment_width = 20
        self.segment_height = 20
        self.position = []
        self.position.append([self.x, self.y])
        self.apple = 1
        self.eaten = False
        self.x_change = 20
        self.y_change = 0

    # Update the snake position
    def get_pos(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.apple > 1:
                for i in range(0, self.apple - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    # Check for collision with the walls
    def check_bounds(self, game):
        if (
            self.x < self.segment_width
            or self.x > game.game_width - 40
            or self.y < self.segment_height
            or self.y > game.game_height - 40
        ):
            return True
        else:
            return False

    # check if snake collides with itself
    def collision_self(self, x, y):
        for i in range(self.apple - 1):
            if self.position[i][0] == x and self.position[i][1] == y:
                return True
        return False

    # If the snake eats an apple, increase the length of the snake
    def eat_apple(self):
        self.position.append([self.x, self.y])
        self.eaten = False
        self.apple = self.apple + 1

    # Play a move each tick
    def play_step(self, move, x, y, game, apple, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.eat_apple()

        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif (
            np.array_equal(move, [0, 1, 0]) and self.y_change == 0
        ):  # right - going horizontal
            move_array = [0, self.x_change]
        elif (
            np.array_equal(move, [0, 1, 0]) and self.x_change == 0
        ):  # right - going vertical
            move_array = [-self.y_change, 0]
        elif (
            np.array_equal(move, [0, 0, 1]) and self.y_change == 0
        ):  # left - going horizontal
            move_array = [0, -self.x_change]
        elif (
            np.array_equal(move, [0, 0, 1]) and self.x_change == 0
        ):  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.check_bounds(game):
            print("Crash")
            game.crash = True

        if self.collision_self(self.x, self.y):
            print("Collision with self")
            game.crash = True

        game.eat(self, apple)

        self.get_pos(self.x, self.y)

    # Render the snake
    def draw_snake(self, x, y, apple, game, options):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(apple):
                self.render.draw_blocks(
                    game.surface,
                    options,
                    pygame.Rect(
                        self.position[len(self.position) - 1 - i][0],
                        self.position[len(self.position) - 1 - i][1],
                        self.segment_width,
                        self.segment_height,
                    ),
                )
            # update_screen()
            pygame.display.update()
        else:
            pygame.time.wait(300)
