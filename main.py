import torch
import transforms
from datasets import VideoLabelDataset
import torchvision
from model.vivit import ViViT
from trainer.trainer import ViVitTrainer
import pytorch_lightning as pl
import torchinfo

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
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
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
dataiter = iter(training_loader)
videos, labels = next(dataiter)
print(f"Images shape : {videos.shape}")

# Code model 
model = ViViT(image_size=224, patch_size=16, num_classes=8, num_frames=30).to(device)
# x = torch.randn((5, 3, 30, 224, 224)).permute(0, 2, 1, 3, 4)
# x = torch.randn((5, 3, 30, 224, 224))
# y = model(x)
# torchinfo.summary(model, input_size = (5, 3, 30, 224, 224), col_names=['input_size', 'output_size', 'num_params'])
vivit = ViVitTrainer(model, weight_decay=0.01, lr = 0.001, epochs=2)
vivit.fit(training_loader, testing_loader, device, 10)
# Start training
# trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices= 1, log_every_n_steps=1)
# trainer.fit(vivit, train_dataloaders=data_loader)