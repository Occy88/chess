from __future__ import annotations

import abc
import math
from enum import Enum
from typing import List
import numpy as np
from numpy.typing import NDArray


class Board:
    def __init__(self, size: NDArray[int, int] = np.array([8, 8])):
        self.board = np.zeros(size)

    def free(self, pos: NDArray[int, int]):
        return self.board[pos] == 0

    def validate_pos(self, pos: NDArray[int, int]):
        return self.board.shape >= pos


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
