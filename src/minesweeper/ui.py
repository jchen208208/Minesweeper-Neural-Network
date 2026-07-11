from __future__ import annotations

import argparse

import pygame

from minesweeper.board import GameStatus
from minesweeper.env import MinesweeperEnv


CELL_SIZE = 40
STATUS_HEIGHT = 56
GRID_LINE = 2
FPS = 60

COLORS = {
    "background": (30, 34, 42),
    "cell_hidden": (88, 96, 112),
    "cell_hover": (108, 118, 138),
    "cell_revealed": (190, 196, 206),
    "grid": (25, 28, 35),
    "text": (238, 241, 245),
    "dark_text": (25, 28, 35),
    "flag": (226, 91, 91),
    "mine": (28, 30, 36),
    "loss": (222, 81, 81),
    "win": (88, 184, 126),
}

NUMBER_COLORS = {
    1: (39, 103, 216),
    2: (39, 148, 74),
    3: (211, 63, 63),
    4: (86, 71, 173),
    5: (150, 64, 49),
    6: (40, 145, 145),
    7: (30, 30, 30),
    8: (90, 90, 90),
}


class MinesweeperUI:
    def __init__(self, width: int, height: int, mines: int, seed: int | None) -> None:
        pygame.init()
        pygame.display.set_caption("Minesweeper DDQN Environment")

        self.env = MinesweeperEnv(width=width, height=height, mine_count=mines, seed=seed)
        self.seed = seed
        self.screen_width = width * CELL_SIZE
        self.screen_height = height * CELL_SIZE + STATUS_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.last_reward = 0.0

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

    def reset(self) -> None:
        self.env.reset(self.seed)
        self.last_reward = 0.0

    def handle_click(self, event: pygame.event.Event) -> None:
        position = event.pos
        if position[1] >= self.env.height * CELL_SIZE:
            return

        col = position[0] // CELL_SIZE
        row = position[1] // CELL_SIZE
        if not self.env.board.in_bounds(row, col):
            return

        if event.button == 1:
            action = self.env.encode_action(row, col)
            _, reward, _, _ = self.env.step(action)
            self.last_reward = reward
        elif event.button == 3:
            self.env.board.toggle_flag(row, col)

    def draw(self) -> None:
        self.screen.fill(COLORS["background"])
        hover_cell = self._hover_cell()

        for row in range(self.env.height):
            for col in range(self.env.width):
                self.draw_cell(row, col, hover_cell == (row, col))

        self.draw_status()
        pygame.display.flip()

    def draw_cell(self, row: int, col: int, hovered: bool) -> None:
        rect = pygame.Rect(
            col * CELL_SIZE,
            row * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        board = self.env.board
        show_mine = board.status == GameStatus.LOST and board.is_mine(row, col)

        if board.is_revealed(row, col) or show_mine:
            color = COLORS["loss"] if show_mine else COLORS["cell_revealed"]
        elif hovered and board.status == GameStatus.ONGOING:
            color = COLORS["cell_hover"]
        else:
            color = COLORS["cell_hidden"]

        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, COLORS["grid"], rect, GRID_LINE)

        if show_mine:
            self._draw_mine(rect)
        elif board.is_flagged(row, col):
            self._draw_flag(rect)
        elif board.is_revealed(row, col):
            count = board.counts[row][col]
            if count > 0:
                self._draw_centered_text(str(count), rect, NUMBER_COLORS[count])

    def draw_status(self) -> None:
        board = self.env.board
        y = self.env.height * CELL_SIZE
        status_rect = pygame.Rect(0, y, self.screen_width, STATUS_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["background"], status_rect)

        if board.status == GameStatus.WON:
            message = "You won. Press R to reset."
            color = COLORS["win"]
        elif board.status == GameStatus.LOST:
            message = "Mine hit. Press R to reset."
            color = COLORS["loss"]
        else:
            message = "Left click: reveal    Right click: flag    R: reset"
            color = COLORS["text"]

        mines_left = max(0, board.mine_count - len(board.flags))
        detail = f"Mines left: {mines_left}    Last reward: {self.last_reward:.2f}"

        message_surface = self.small_font.render(message, True, color)
        detail_surface = self.small_font.render(detail, True, COLORS["text"])
        self.screen.blit(message_surface, (12, y + 8))
        self.screen.blit(detail_surface, (12, y + 30))

    def _hover_cell(self) -> tuple[int, int] | None:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_y >= self.env.height * CELL_SIZE:
            return None
        row = mouse_y // CELL_SIZE
        col = mouse_x // CELL_SIZE
        if self.env.board.in_bounds(row, col):
            return row, col
        return None

    def _draw_centered_text(
        self, text: str, rect: pygame.Rect, color: tuple[int, int, int]
    ) -> None:
        surface = self.font.render(text, True, color)
        text_rect = surface.get_rect(center=rect.center)
        self.screen.blit(surface, text_rect)

    def _draw_flag(self, rect: pygame.Rect) -> None:
        pole_start = (rect.left + CELL_SIZE // 3, rect.top + 10)
        pole_end = (rect.left + CELL_SIZE // 3, rect.bottom - 8)
        flag_points = [
            pole_start,
            (rect.right - 10, rect.top + 15),
            (rect.left + CELL_SIZE // 3, rect.top + 22),
        ]
        pygame.draw.line(self.screen, COLORS["dark_text"], pole_start, pole_end, 3)
        pygame.draw.polygon(self.screen, COLORS["flag"], flag_points)

    def _draw_mine(self, rect: pygame.Rect) -> None:
        pygame.draw.circle(self.screen, COLORS["mine"], rect.center, CELL_SIZE // 5)
        pygame.draw.circle(self.screen, COLORS["text"], rect.center, CELL_SIZE // 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Minesweeper.")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MinesweeperUI(args.width, args.height, args.mines, args.seed).run()


if __name__ == "__main__":
    main()
