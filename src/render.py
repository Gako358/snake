import pygame
from config import Options


class Render:
    def __init__(self):
        self.options = Options()

    # Draw status bar
    def draw_text(self, game, score, record):
        font = pygame.font.SysFont("Segoe UI", 19)
        bold_font = pygame.font.SysFont("Segoe UI", 19, True)
        text_score = font.render("SCORE: ", True, self.options.colors["STATUS"])
        text_score_number = font.render(str(score), True, self.options.colors["STATUS"])
        text_highest = font.render(
            "HIGHEST SCORE: ", True, self.options.colors["STATUS"]
        )
        text_highest_number = bold_font.render(
            str(record), True, self.options.colors["MAGENTA"]
        )
        game.surface.blit(text_score, (45, game.game_height))
        game.surface.blit(text_score_number, (120, game.game_height))
        game.surface.blit(text_highest, (190, game.game_height))
        game.surface.blit(text_highest_number, (350, game.game_height))

    # Draw game over screen
    def display(self, snake, apple, game, record):
        game.surface.fill(self.options.colors["BACKGROUND"])
        self.draw_text(game, game.score, record)
        snake.draw_snake(
            snake.position[-1][0],
            snake.position[-1][1],
            snake.apple,
            game,
            self.options.colors["GREEN"],
        )
        apple.draw_apple(apple.x_pos, apple.y_pos, game, self.options.colors["RED"])

    # Helper function to draw blocks
    def draw_blocks(self, surface, color, rect):
        pygame.draw.rect(surface, color, rect)

    # Update the screen
    def update_screen(self):
        pygame.display.update()
