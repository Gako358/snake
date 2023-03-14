import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Options:
    def __init__(self):
        self.colors = {
            "WHITE": (235, 219, 178),
            "RED": (251, 73, 52),
            "BLUE": (131, 165, 152),
            "BLACK": (40, 40, 40),
            "BACKGROUND": (235, 219, 178),
            "STATUS": (80, 73, 69),
            "GREEN": (152, 151, 26),
            "YELLOW": (215, 153, 33),
            "MAGENTA": (177, 98, 134),
            "AQUA": (104, 157, 106),
        }
        self.game_width = 440
        self.game_height = 440
        self.layers = [256, 73, 46]
        self.learning_rate = 0.0001
        self.num_games = 190  # number of games to train, not increasing after 190 on single hidden layer
        self.memory_size = 10000
        self.batch_size = 1000
        self.weights_path = "./src/weights/weights.h5"
        # self.weights_path = "./weights/weights.h5"
        self.plot_path = "./src/plots/plot.png"
        # self.plot_path = "./plots/plot.png"
        self.delay = 40

    def plot(self, num_games, score, time_start, time_end):
        sns.set(color_codes=True, font_scale=1.5)
        sns.set_style("white")
        plt.figure(figsize=(13, 8))
        ax = sns.regplot(
            np.array([num_games])[0],
            np.array([score])[0],
            x_jitter=0.1,
            label="Data",
            fit_reg=False,
        )
        total_time = round(time_end - time_start, 2)
        y_mean = [np.mean(score)] * len(num_games)
        ax.plot(num_games, y_mean, label="Mean", linestyle="--")
        ax.legend(loc="upper right")
        ax.set(
            xlabel="# games - Time to train: " + str(total_time) + " seconds",
            ylabel="score",
        )
        plt.savefig(self.plot_path)

    def dec_num_games(self):
        self.num_games -= 1

    def inc_num_games(self):
        self.num_games += 1
