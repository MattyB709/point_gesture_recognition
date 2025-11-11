import torch
from .pointing_dataset import PointingDataset 
from torch.utils.data import DataLoader

def train(num_epochs, batch_size):
    train_set = PointingDataset("data/")
    train_loader = DataLoader(train_set, batch_size = batch_size)




