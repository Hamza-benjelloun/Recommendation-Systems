import torch

from src.model.Model import RecSysModel
from src.utils.MovieDataset import MovieDataset
from src.utils.utils import get_data_loader, encode_data, get_train_test_split,get_optimizer, get_loss_function, get_scheduler,train, test

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

if __name__ == "__main__":
    pass