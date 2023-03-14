import pygame
import functools


class Menu:
    def __init__(self, options, game_width, game_height):
        self.options = options
        self.surface = pygame.display.set_mode((game_width, game_height))
        self.font = pygame.font.SysFont("Arial", 30)
        self.small_font = pygame.font.SysFont("Arial", 20)

    def text_object(self, text, font, color):
        text_surface = font.render(text, True, color)
        return text_surface, text_surface.get_rect()

    def button(self, msg, x, y, w, h, ic, ac, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x + w > mouse[0] > x and y + h > mouse[1] > y:
            pygame.draw.rect(self.surface, ac, (x, y, w, h))
            if click[0] == 1 and action is not None:
                action()
        else:
            pygame.draw.rect(self.surface, ic, (x, y, w, h))
        small_text = pygame.font.SysFont("Arial", 20)
        text_surf, text_rect = self.text_object(
            msg, small_text, self.options.colors["STATUS"]
        )
        text_rect.center = ((x + (w / 2)), (y + (h / 2)))
        self.surface.blit(text_surf, text_rect)

    def intro(self, play, model):
        intro = True
        while intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()

            self.surface.fill(self.options.colors["BACKGROUND"])
            text_surface, text_rect = self.text_object(
                "Snake", self.font, self.options.colors["STATUS"]
            )
            text_rect.center = (
                self.options.game_width / 2,
                self.options.game_height / 2,
            )
            self.surface.blit(text_surface, text_rect)
            num_games, num_rect = self.text_object(
                "Games to train: " + str(self.options.num_games),
                self.small_font,
                self.options.colors["STATUS"],
            )
            num_rect.center = (
                self.options.game_width / 2,
                self.options.game_height / 2 + 50,
            )
            self.surface.blit(num_games, num_rect)

            info, info_rect = self.text_object(
                "Press ESC to quit",
                self.small_font,
                self.options.colors["STATUS"],
            )
            info_rect.center = (
                self.options.game_width / 2,
                self.options.game_height / 2 + 100,
            )
            self.surface.blit(info, info_rect)

            self.button(
                "Play",
                self.options.game_width / 2 - 200,
                400,
                100,
                50,
                self.options.colors["GREEN"],
                self.options.colors["MAGENTA"],
                functools.partial(play, False),
            )
            self.button(
                "Training",
                self.options.game_width - 120,
                400,
                100,
                50,
                self.options.colors["GREEN"],
                self.options.colors["MAGENTA"],
                functools.partial(play, True),
            )
            self.button(
                "-",
                self.options.game_width / 2 - 50,
                400,
                50,
                50,
                self.options.colors["RED"],
                self.options.colors["YELLOW"],
                self.options.dec_num_games,
            )
            self.button(
                "+",
                self.options.game_width / 2 + 30,
                400,
                50,
                50,
                self.options.colors["BLUE"],
                self.options.colors["YELLOW"],
                self.options.inc_num_games,
            )

            self.button(
                "Model",
                self.options.game_width - 120,
                370,
                100,
                30,
                self.options.colors["AQUA"],
                self.options.colors["YELLOW"],
                model,
            )
            pygame.display.update()
