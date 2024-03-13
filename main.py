import torch
import transforms
from datasets import VideoLabelDataset
import torchvision
from model.vivit import ViViT
from trainer.trainer import ViVitTrainer
import pytorch_lightning as pl
# Load video dataset
dataset = VideoLabelDataset(
    "data.csv",
    transform=torchvision.transforms.Compose([
        transforms.VideoFilePathToTensor(max_len=30, padding_mode='last', fps=10),
        transforms.VideoCenterCrop(size = (1080, 640)),
        transforms.VideoRandomHorizontalFlip(0.5),
        transforms.VideoResize([224, 224]),
    ])
)
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 5)
print("Done")
# Code model 
model = ViViT(image_size=224, patch_size=16, num_classes=8, num_frames=30)
x = torch.randn((1, 30, 3, 224, 224))
y = model(x)
vivit = ViVitTrainer(model)
print("Done 2")
# Start training
trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, log_every_n_steps=1)
trainer.fit(vivit, train_dataloaders=data_loader)