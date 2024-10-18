import math
import os
import sys

sys.path.append(os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "..", ".."))

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from fuzzywuzzy import process
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from mario.block2vec.block2vec_dataset import Block2VecDataset
from mario.block2vec.skip_gram_model import SkipGramModel
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

class Block2Vec(pl.LightningModule):
    def __init__(self, opt, level=None):
        super().__init__()
        self.args = opt
        self.save_hyperparameters()
        self.dataset = Block2VecDataset(
            self.args.input_dir,
            self.args.input_name if level is None else "",
            neighbor_radius=self.args.neighbor_radius,
            level=level
        )
        self.emb_size = len(self.dataset.block2idx)
        self.model = SkipGramModel(self.emb_size, self.args.emb_dimension)
        self.textures = dict()
        self.learning_rate = self.args.initial_lr

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        loss = self.forward(*batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            math.ceil(len(self.dataset) / self.args.batch_size) *
            self.args.epochs,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )

    def on_train_epoch_end(self):
        self.save_embedding(
            self.dataset.idx2block, self.args.output_dir
        )
        self.create_confusion_matrix(
            self.dataset.idx2block, self.args.output_dir)

    def save_embedding(self, id2block: Dict[int, str], output_path: str):
        embeddings = self.model.target_embeddings.weight
        # embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().data.numpy()
        embedding_dict = {}
        with open(os.path.join(output_path, "embeddings.txt"), "w") as f:
            f.write("%d %d\n" % (len(id2block), self.args.emb_dimension))
            for wid, w in id2block.items():
                e = " ".join(map(lambda x: str(x), embeddings[wid]))
                embedding_dict[w] = torch.from_numpy(embeddings[wid])
                f.write("%s %s\n" % (w, e))
        np.save(os.path.join(output_path, "embeddings.npy"), embeddings)
        with open(os.path.join(output_path, f"representations.pkl"), "wb") as f:
            pickle.dump(embedding_dict, f)
        return embedding_dict

    def create_confusion_matrix(self, id2block: Dict[int, str], output_path: str):
        rcParams.update({"font.size": 6})
        names = []
        dists = np.zeros((len(id2block), len(id2block)))
        for i, b1 in id2block.items():
            names.append(b1)
            for j, b2 in id2block.items():
                dists[i, j] = F.mse_loss(
                    self.model.target_embeddings.weight.data[i],
                    self.model.target_embeddings.weight.data[j],
                )
        confusion_display = ConfusionMatrixDisplay(dists, display_labels=names)
        confusion_display.plot(include_values=False,
                               xticks_rotation="vertical")
        confusion_display.ax_.set_xlabel("")
        confusion_display.ax_.set_ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "dist_matrix.png"))
        plt.close()