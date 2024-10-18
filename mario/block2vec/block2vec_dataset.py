import os
import sys

sys.path.append(os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "..", ".."))

from collections import defaultdict
from itertools import product
from typing import Tuple

from loguru import logger
import numpy as np
from tqdm import tqdm
from mario.tokens import REPLACE_TOKENS

from torch.utils.data.dataset import Dataset


class Block2VecDataset(Dataset):

    def __init__(self, input_dir: str, input_name: str, neighbor_radius: int = 1, level=None):
        """Block dataset with configurable neighborhood radius.

        Args:
            input_name (str): path to the Mario level
            neighbor_radius (int): neighbors to retrieve as a context
        """
        super().__init__()
        self.input_name = input_name

        if level is None:
            self.level = load_level_from_text(os.path.join(input_dir, self.input_name))
        else:
            self.level = level

        padding = 2 * neighbor_radius  # one token on each side

        self.x_lims = (0, len(self.level))
        self.y_lims = (0, len(self.level[-1]))
        self.x_dim = self.x_lims[1] - self.x_lims[0] - padding
        self.y_dim = self.y_lims[1] - self.y_lims[0] - padding
        logger.info("Cutting {} x {} volume from {}", self.x_dim,
                    self.y_dim, self.input_name)
        self.neighbor_radius = neighbor_radius
        self._read_blocks()
        self._init_discards()

    def _init_discards(self):
        t = 0.001
        token_frequencies = list(self.block_frequency.values())
        f = np.array(token_frequencies) / sum(token_frequencies)
        self.discards = 1.0 - (np.sqrt(f / t) + 1) * (t / f)

    def _read_blocks(self):
        self.block_frequency = defaultdict(int)
        coordinates = [(x, y) for x, y in product(range(self.x_lims[0], self.x_lims[1]),
                                                        range(self.y_lims[0], self.y_lims[1]))]
        logger.info("Collecting {} blocks", len(self))
        for name in tqdm([self._get_block(*coord) for coord in coordinates]):
            self.block_frequency[name] += 1
        logger.info(
            "Found the following blocks {blocks}", blocks=dict(self.block_frequency))
        self.block2idx = dict()
        self.idx2block = dict()
        for name, count in self.block_frequency.items():
            block_idx = len(self.block2idx)
            self.block2idx[name] = block_idx
            self.idx2block[block_idx] = name

    def __getitem__(self, index):
        coords = self._idx_to_coords(index)
        block = self._get_block(*coords)
        target = self.block2idx[block]
        if np.random.rand() < self.discards[target]:
            return self.__getitem__(np.random.randint(self.__len__()))
        neighbor_blocks = self._get_neighbors(*coords)
        context = np.array([self.block2idx[n] for n in neighbor_blocks])
        return target, context

    def _idx_to_coords(self, index):
        z = index
        y = int(((index - z)) % (self.y_dim + 1))
        x = int(((index - z) - y) / (self.y_dim + 1))
        x += self.x_lims[0] + self.neighbor_radius
        y += self.y_lims[0] + self.neighbor_radius
        return x, y

    def _get_block(self, x, y):
        block = self.level[x][y]
        return block

    def _get_neighbors(self, x, y):
        neighbor_coords = [(x + x_diff, y + y_diff) for x_diff, y_diff in product(list(
            range(-self.neighbor_radius, self.neighbor_radius + 1)), repeat=2) if x_diff != 0 or y_diff != 0]
        return [self._get_block(*coord) for coord in neighbor_coords]

    def __len__(self):
        return self.x_dim * self.y_dim
    
def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level