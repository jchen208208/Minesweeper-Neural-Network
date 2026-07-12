from __future__ import annotations

import argparse

import pygame

from minesweeper.board import GameStatus
from minesweeper.env import MinesweeperEnv


CELL_SIZE = 40
STATUS_HEIGHT = 98
GRID_LINE = 2
FPS = 60
MIN_WIDTH = 5
MAX_WIDTH = 40
MIN_HEIGHT = 5
MAX_HEIGHT = 30

# name -> (width, height, mines)
DIFFICULTIES: dict[str, tuple[int, int, int]] = {
    "Beginner": (9, 9, 10),
    "Intermediate": (16, 16, 40),
    "Expert": (30, 16, 99),
}
DIFFICULTY_KEYS = list(DIFFICULTIES)

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
    def __init__(
        self,
        width: int,
        height: int,
        mines: int,
        seed: int | None,
        no_guess: bool = True,
        difficulty: str | None = None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Minesweeper DDQN Environment")

        self.seed = seed
        self.no_guess = no_guess
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.last_reward = 0.0
        self.difficulty = difficulty
        self.set_board(width, height, mines)

    def set_board(self, width: int, height: int, mines: int) -> None:
        self.env = MinesweeperEnv(
            width=width,
            height=height,
            mine_count=mines,
            seed=self.seed,
            no_guess=self.no_guess,
        )
        self.screen_width = width * CELL_SIZE
        self.screen_height = height * CELL_SIZE + STATUS_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.last_reward = 0.0

    def select_difficulty(self, name: str) -> None:
        self.difficulty = name
        width, height, mines = DIFFICULTIES[name]
        self.set_board(width, height, mines)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

    def handle_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_r:
            self.reset()
            return
        preset_keys = {
            pygame.K_1: 0,
            pygame.K_2: 1,
            pygame.K_3: 2,
        }
        if event.key in preset_keys:
            self.select_difficulty(DIFFICULTY_KEYS[preset_keys[event.key]])
            return

        if event.key == pygame.K_LEFT:
            self._resize_board(self.env.width - 1, self.env.height)
        elif event.key == pygame.K_RIGHT:
            self._resize_board(self.env.width + 1, self.env.height)
        elif event.key == pygame.K_UP:
            self._resize_board(self.env.width, self.env.height + 1)
        elif event.key == pygame.K_DOWN:
            self._resize_board(self.env.width, self.env.height - 1)
        elif event.key == pygame.K_LEFTBRACKET:
            self._set_mines(self.env.mine_count - 1)
        elif event.key == pygame.K_RIGHTBRACKET:
            self._set_mines(self.env.mine_count + 1)

    def _resize_board(self, width: int, height: int) -> None:
        width = max(MIN_WIDTH, min(MAX_WIDTH, width))
        height = max(MIN_HEIGHT, min(MAX_HEIGHT, height))
        area = width * height
        old_area = self.env.width * self.env.height
        old_density = self.env.mine_count / old_area
        mines = max(1, min(area - 1, round(old_density * area)))
        self.difficulty = None
        self.set_board(width, height, mines)

    def _set_mines(self, mines: int) -> None:
        area = self.env.width * self.env.height
        mines = max(1, min(area - 1, mines))
        self.difficulty = None
        self.set_board(self.env.width, self.env.height, mines)

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
            message = "L-click reveal   R-click flag   R reset"
            color = COLORS["text"]

        mines_left = max(0, board.mine_count - len(board.flags))
        mode = "no-guess" if self.no_guess else "classic"
        difficulty = self.difficulty or "Custom"
        detail = (
            f"Mines left: {mines_left}   "
            f"{difficulty} {self.env.width}x{self.env.height}   {mode}"
        )
        presets = "Size: 1 Beginner  2 Intermediate  3 Expert"
        custom = "Custom: arrows resize board  [ / ] mines"

        message_surface = self.small_font.render(message, True, color)
        detail_surface = self.small_font.render(detail, True, COLORS["text"])
        presets_surface = self.small_font.render(presets, True, COLORS["text"])
        self.screen.blit(message_surface, (12, y + 6))
        self.screen.blit(detail_surface, (12, y + 28))
        self.screen.blit(presets_surface, (12, y + 50))
        custom_surface = self.small_font.render(custom, True, COLORS["text"])
        self.screen.blit(custom_surface, (12, y + 72))

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
    parser.add_argument(
        "--difficulty",
        choices=DIFFICULTY_KEYS,
        default=None,
        help="Start with a preset board size instead of --width/--height/--mines.",
    )
    parser.add_argument(
        "--allow-guessing",
        action="store_true",
        help="Disable no-guess generation and allow boards that need guessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.difficulty is not None:
        width, height, mines = DIFFICULTIES[args.difficulty]
    else:
        width, height, mines = args.width, args.height, args.mines
    MinesweeperUI(
        width,
        height,
        mines,
        args.seed,
        no_guess=not args.allow_guessing,
        difficulty=args.difficulty,
    ).run()


if __name__ == "__main__":
    main()
