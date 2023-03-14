import pygame
from random import randint
from render import Render


class Apple(object):
    def __init__(self):
        self.render = Render()
        self.x_pos = 240
        self.y_pos = 200
        self.food_width = 20
        self.food_height = 20

    # Get a new position for the apple to be placed, random
    def get_pos(self, game, player):
        new_pos_x = randint(self.food_width, game.game_width - 40)
        self.x_pos = new_pos_x - new_pos_x % 20
        new_pos_y = randint(self.food_height, game.game_height - 40)
        self.y_pos = new_pos_y - new_pos_y % 20
        if [self.x_pos, self.y_pos] not in player.position:
            return self.x_pos, self.y_pos
        else:
            self.get_pos(game, player)

    # Draw the apple
    def draw_apple(self, x, y, game, options):
        self.render.draw_blocks(
            game.surface, options, (x, y, self.food_width, self.food_height)
        )
        pygame.display.update()
