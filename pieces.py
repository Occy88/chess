from __future__ import annotations
from string import ascii_lowercase
import abc
import math
import time
from enum import Enum
from typing import List
import numpy as np
from numpy.typing import NDArray
from blessed import Terminal
from functools import partial

# python 2/3 compatibility, provide 'echo' function as an
# alias for "print withterm newline and flush"
try:
    # pylint: disable=invalid-name
    #         Invalid constant name "echo"
    echo = partial(print, end='', flush=True)
    echo(u'')
except TypeError:
    # TypeError: 'flush' is an invalid keyword argument for this function
    import sys


    def echo(text):
        """Python 2 version of print(end='', flush=True)."""
        sys.stdterm.write(u'{0}'.format(text))
        sys.stdterm.flush()


class Board:
    def __init__(self, size: NDArray[int, int] = np.array([8, 8])):
        self.board = np.zeros(size)

    def free(self, pos: NDArray[int, int]):
        return self.board[pos] == 0

    def validate_pos(self, pos: NDArray[int, int]):
        return self.board.shape >= pos

    def draw_square(self, term: Terminal, coord: NDArray[int, int], color: Color):
        scale = np.array([2, 4], dtype=int)
        coord_scaled = scale * coord
        if color == Color.WHITE:
            col = term.on_sienna
        else:
            col = term.on_peru
        # print(print(coord, coord_scaled))
        for i in range(coord_scaled[0], coord_scaled[0] + scale[0]):
            for j in range(coord_scaled[1], coord_scaled[1] + scale[1]):
                echo(term.move_yx(i, j))
                echo(col(u'\u2655'))
        echo(term.move(coord_scaled[0], coord_scaled[1]))
        echo(col(f'{ascii_lowercase[coord[1]]}{coord[0]}',))

    def draw(self, term: Terminal):
        color_bg = term.on_blue
        echo(term.move_yx(1, 1))
        echo(color_bg(term.clear))
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if i % 2 and not j % 2 or not i % 2 and j % 2:
                    self.draw_square(term, np.array([i, j], dtype=int), Color.WHITE)
                else:
                    self.draw_square(term, np.array([i, j], dtype=int), Color.BLACK)

    def set_piece(self, piece: Piece):
        pass


class Color(Enum):
    WHITE = 1
    BLACK = 0


class Piece:
    def __init__(self, color: Color.WHITE, pos: NDArray[int, int]):
        self.color = color
        self.pos = pos

    def __hash__(self):
        return self.pos

    @abc.abstractmethod
    def valid_moves(self, board: Board) -> List[NDArray[int, int]]:
        """
        Expand
        :param board:
        :return:
        """
        ...


def expand_moves(board: Board, coord: NDArray[int, int], moves: list[NDArray[int, int]], n):
    exp = []
    for m in moves:
        c = coord + m
        while board.validate_pos(coord) and n > 0:
            exp.append(c)
            if not board.free(c):
                break
            n -= 1
            c += m
    return exp


def free_diagonal_indices(board: Board, coord: NDArray[int, int], n=math.inf):
    moves = [np.array(-1, -1),
             np.array(-1, 1),
             np.array(1, -1),
             np.array(1, 1)]
    return expand_moves(board, coord, moves, n)


def free_horizontal_indices(board: Board, coord: NDArray[int, int], n=math.inf):
    moves = [np.array(0, -1),
             np.array(-1, 0),
             np.array(1, 0),
             np.array(0, 1)]
    return expand_moves(board, coord, moves, n)


def free_knight_indices(board: Board, coord: NDArray[int, int], n=math.inf):
    moves = [
        np.array(-1, 2),
        np.array(-1, -2),
        np.array(1, 2),
        np.array(1, -2),
        np.array(-2, 1),
        np.array(-2, -1),
        np.array(2, 1),
        np.array(2, -1),

    ]
    return expand_moves(board, coord, moves)


class Bishop(Piece):
    def valid_moves(self, board: Board) -> List[NDArray[int, int]]:
        return free_diagonal_indices(board, self.pos)


class Knight(Piece):
    def valid_moves(self, board: Board) -> List[NDArray[int, int]]:
        return free_knight_indices(board, self.pos)


class Queen(Piece):
    def valid_moves(self, board: Board) -> List[NDArray[int, int]]:
        return free_diagonal_indices(board, self.pos) + free_horizontal_indices(board, self.pos)


class Rook(Piece):
    def valid_moves(self, board: Board) -> List[NDArray[int, int]]:
        return free_horizontal_indices(board, self.pos)


class King(Piece):
    def valid_moves(self, board: Board) -> List[NDArray[int, int]]:
        return free_diagonal_indices(board, self.pos, 1) + free_horizontal_indices(board, self.pos, 1)


class Game:
    def __init__(self, board=Board()):
        self.term = Terminal()
        self.board = board

    def draw(self):
        inp = None
        with self.term.hidden_cursor(), self.term.cbreak(), self.term.location():
            while inp not in (u'q', u'Q'):
                # inp = self.term.inkey()
                self.board.draw(self.term)
                time.sleep(10)


g = Game()
g.draw()
