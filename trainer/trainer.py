import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

class ViVitTrainer(pl.LightningModule):
    def __init__(self, model):
        super(ViVitTrainer, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        labels_hat = torch.argmax(y_pred, dim=1)
        train_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'train_loss': loss, 'train_acc': train_acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
          y_pred = self(x)
        loss = self.loss(y_pred, y)
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc}, prog_bar=True)