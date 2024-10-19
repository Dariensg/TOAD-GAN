import torch
from loguru import logger
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import load_pkl
from mario.block2vec.block2vec import Block2Vec

from .tokens import TOKEN_GROUPS, REPLACE_TOKENS


# Miscellaneous functions to deal with ascii-token-based levels.


def group_to_token(tensor, tokens, token_groups=TOKEN_GROUPS):
    """ Converts a token group level tensor back to a full token level tensor. """
    new_tensor = torch.zeros(tensor.shape[0], len(tokens), *tensor.shape[2:]).to(
        tensor.device
    )
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, i] = tensor[:, group_idx]
                break
    return new_tensor


def token_to_group(tensor, tokens, token_groups=TOKEN_GROUPS):
    """ Converts a full token tensor to a token group tensor. """
    new_tensor = torch.zeros(tensor.shape[0], len(token_groups), *tensor.shape[2:]).to(
        tensor.device
    )
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, group_idx] += tensor[:, i]
                break
    return new_tensor


def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level


def ascii_to_one_hot_level(level, tokens):
    """ Converts an ascii level to a full token level tensor. """
    oh_level = torch.zeros((len(tokens), len(level), len(level[-1])))
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in tokens and token != "\n":
                oh_level[tokens.index(token), i, j] = 1
    return oh_level

def ascii_to_block2vec_level(level, dim, repr):
    oh_level = torch.zeros((dim, len(level), len(level[-1])))

    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in repr.keys() and token != "\n":
                oh_level[:, i, j] = repr[token]

    return oh_level

def encoded_to_ascii_level(level, tokens, block2repr, repr_type):
    if repr_type == 'block2vec':
        return block2vec_to_ascii_level(level, block2repr)
    else:
        return one_hot_to_ascii_level(level, tokens)


def one_hot_to_ascii_level(level, tokens):
    """ Converts a full token level tensor to an ascii level. """
    ascii_level = []
    for i in range(level.shape[2]):
        line = ""
        for j in range(level.shape[3]):
            line += tokens[level[:, :, i, j].argmax()]
        if i < level.shape[2] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level

def block2vec_to_ascii_level(level, block2repr):
    # Convert block2vec level into 2D array of block indices
    block_embs = torch.stack(list(block2repr.values()))
    world_vec = level.squeeze().permute(1,2,0).unsqueeze(3)
    blocks_vec = block_embs.permute(1,0)[None,None,...]
    dist = (world_vec - blocks_vec).pow(2).sum(dim=-2)
    labels = dist.argmin(dim=-1)

    ascii_level = []
    for i in range(labels.shape[0]):
        line = ""
        for j in range(labels.shape[1]):
            line += list(block2repr.keys())[labels[i, j]]
        if i < labels.shape[0] - 1:
            line += "\n"
        ascii_level.append(line)

    return ascii_level


def get_all_tokens(opt, replace_tokens=REPLACE_TOKENS):
    tokens = set()

    # Get D1 tokens
    d1_txt_level = load_level_from_text("%s/%s" % (opt.input_dir, opt.d1_input_name), replace_tokens)
    for line in d1_txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n" and token not in replace_tokens.items():
                tokens.add(token)

    # Get D2 tokens
    d2_txt_level = load_level_from_text("%s/%s" % (opt.input_dir, opt.d2_input_name), replace_tokens)
    for line in d2_txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n" and token not in replace_tokens.items():
                tokens.add(token)

    tokens = list(tokens)
    tokens.sort()  # necessary! otherwise we won't know the token order later

    return tokens


def read_level(opt, input_name, repr=None, tokens=None, replace_tokens=REPLACE_TOKENS):
    """ Wrapper function for read_level_from_file using namespace opt. Updates parameters for opt."""
    level, uniques = read_level_from_file(opt.input_dir, input_name, repr, opt.repr_type, tokens, replace_tokens)
    opt.token_list = uniques if tokens == None else tokens
    logger.info("Tokens in level {}", opt.token_list)
    opt.nc_current = len(uniques) if tokens == None else len(tokens)
    return level


def read_level_from_file(input_dir, input_name, repr, repr_type, tokens=None, replace_tokens=REPLACE_TOKENS):
    """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token. """
    txt_level = load_level_from_text("%s/%s" % (input_dir, input_name), replace_tokens)

    if repr_type == 'block2vec' and repr is not None:
        uniques = [u for u in repr.keys()]
        dim = len(repr[uniques[0]])

        oh_level = ascii_to_block2vec_level(txt_level, dim, repr)
    else:
        uniques = set()
        for line in txt_level:
            for token in line:
                # if token != "\n" and token != "M" and token != "F":
                if token != "\n" and token not in replace_tokens.items():
                    uniques.add(token)
        uniques = list(uniques)
        uniques.sort()  # necessary! otherwise we won't know the token order later


        oh_level = ascii_to_one_hot_level(txt_level, uniques if tokens is None else tokens)


    return oh_level.unsqueeze(dim=0), uniques


def place_a_mario_token(level):
    """ Finds the first plausible spot to place Mario on. Especially important for levels with floating platforms.
    level is expected to be ascii."""
    # First check if default spot is available
    for j in range(1, 4):
        if level[-3][j] == '-' and level[-2][j] in ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']:
            tmp_slice = list(level[-3])
            tmp_slice[j] = 'M'
            level[-3] = "".join(tmp_slice)
            return level

    # If not, check for first possible location from left
    for j in range(len(level[-1])):
        for i in range(1, len(level)):
            if level[i - 1][j] == '-' and level[i][j] in ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']:
                tmp_slice = list(level[i - 1])
                tmp_slice[j] = 'M'
                level[i - 1] = "".join(tmp_slice)
                return level

    return level  # Will only be reached if there is no place to put Mario

def train_block2vec(opt, all_tokens, replace_tokens=REPLACE_TOKENS):
    d1_real = read_level(opt, opt.d1_input_name, tokens=all_tokens, replace_tokens=replace_tokens).to(opt.device)
    d2_real = read_level(opt, opt.d2_input_name, tokens=all_tokens, replace_tokens=replace_tokens).to(opt.device)

    training_real = torch.cat((d1_real, d2_real), dim=-1)

    training_real = one_hot_to_ascii_level(training_real, all_tokens)


    logger.info("Training block2vec...")
    block2vec = Block2Vec(opt, level=training_real)
    trainer = pl.Trainer(accelerator="cpu", devices="auto", max_epochs=opt.epochs, fast_dev_run=opt.debug)
    trainer.fit(block2vec)

    block2repr = load_pkl("representations",
                        f"./output/block2vec/")
    
    opt.block2repr = block2repr
    opt.nc_current = opt.emb_dimension

    return block2repr