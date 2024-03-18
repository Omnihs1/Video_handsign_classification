import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
import numpy as np
from tqdm import tqdm 

class ViVitTrainer():
    def __init__(self, model, weight_decay, lr, epochs):
        super(ViVitTrainer, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), weight_decay = weight_decay, lr = lr)
        self.epochs = epochs
    def forward(self, x):
        return self.model(x)

    def fit(self, train_dataloader, test_dataloader, device):
        self.model.train(True)
        train_loss = []
        test_loss = []
        for i in range(self.epochs):
            avg_loss = self.train_one_epoch(train_dataloader, device)
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            # running_vloss = 0.
            acc = 0
            with torch.no_grad():
                for i, vdata in enumerate(test_dataloader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = self.model(vinputs)
                    metric = MulticlassAccuracy(num_classes=8).to(device)
                    acc = metric(voutputs, vlabels)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            # avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, running_vloss))
            print('ACC valid {}'.format(acc))
            # Sample list of numbers
            train_loss.append(avg_loss.cpu())
            test_loss.append(running_vloss.cpu())
        # print(f"Loss {running_vloss}")
        # Convert the list to a NumPy array
        train_loss = np.array(train_loss)
        test_loss = np.array(test_loss)
        # Save the array to a .npy file
        np.save('train_loss.npy', train_loss)
        np.save('test_loss.npy', test_loss)
        torch.save(self.model.state_dict(), "model.pt")

    def train_one_epoch(self, dataloader, device):
        running_loss = 0.
        last_loss = 0.
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(dataloader)):
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