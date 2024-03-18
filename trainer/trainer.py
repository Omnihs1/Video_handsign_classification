import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

class ViVitTrainer():
    def __init__(self, model, weight_decay, lr, epochs):
        super(ViVitTrainer, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), weight_decay = weight_decay, lr = lr)
        self.epochs = epochs
    def forward(self, x):
        return self.model(x)

    def fit(self, train_dataloader, test_dataloader, device, epochs = 1):
        self.model.train(True)
        for i in range(self.epochs):
            avg_loss = self.train_one_epoch(train_dataloader, device)
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            # running_vloss = 0.
            with torch.no_grad():
                for i, vdata in enumerate(test_dataloader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

        # avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, running_vloss))
        # print(f"Loss {running_vloss}")
        torch.save(self.model.state_dict(), "model.pt")

    def train_one_epoch(self, dataloader, device):
        running_loss = 0.
        last_loss = 0.
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs.to(torch.device('cuda:0'))
            # Zero your gradients for every batch!
            self.optim.zero_grad()
            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optim.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 5 == 0:
                last_loss = running_loss / 5 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                # tb_x = epoch_index * len(training_loader) + i + 1
                # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss