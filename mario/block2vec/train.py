import os
import sys

sys.path.append(os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "..", ".."))

from config import get_arguments, post_config
from mario.block2vec.block2vec import Block2Vec
from utils import load_pkl

from typing import Tuple

import pytorch_lightning as pl

def main():
    # Parse arguments
    parser = get_arguments()
    parser.add_argument("--input-name", type=str, help="input to train embedding on", default="lvl_1-1.txt")
    parser.add_argument("--output-dir", type=str, help="output path for embedding", default=os.path.join(
        os.path.dirname(__file__),
            "..",
            "..",
            "output",
            "block2vec",
        ))
    parser.add_argument("--emb-dimension", type=int, help="number of dimensions in embedding", default=3)
    parser.add_argument("--epochs", type=int, help="number of epochs to train", default=30)
    parser.add_argument("--batch-size", type=int, help="training batch size", default=256)
    parser.add_argument("--initial-lr", type=int, help="initial learning rate", default=1e-3)
    parser.add_argument("--neighbor-radius", type=int, help="radius of neighbors to use for training", default=1)

    parser.add_argument("--debug", type=bool, help="enable debugging mode", default=False)

    opt = parser.parse_args()

    opt = post_config(opt)

    block2vec = Block2Vec(opt)
    trainer = pl.Trainer(accelerator="cpu", devices="auto", max_epochs=opt.epochs, fast_dev_run=opt.debug)
    trainer.fit(block2vec)


if __name__ == "__main__":
    main()