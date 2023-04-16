import torch

import pandas as pd
from src.model.Model import RecSysModel
from src.utils.MovieDataset import MovieDataset
from src.utils.DataUtils import (
    get_data_loader,
    encode_data,
    get_train_test_split,
    get_optimizer,
    get_loss_function,
    get_scheduler,
    train,
    test,
)
import configparser
import logging


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    data_path = config["DATA"]["PATH"]

    shuffle = bool(config["LOADER"]["SHUFFLE"])
    batch_size = int(config["LOADER"]["BATCH_SIZE"])
    num_workers = int(config["LOADER"]["NUM_WORKERS"])

    step_size = int(config["SCHEDULER"]["STEP_SIZE"])
    gamma = float(config["SCHEDULER"]["GAMMA"])

    optimizer = config["MODEL"]["OPTIMIZER"].lower()
    loss_function = config["MODEL"]["LOSS_FUNCTION"].lower()
    epochs = int(config["MODEL"]["EPOCHS"])

    # load data
    df = pd.read_csv(data_path)

    # encode data
    df = encode_data(df)

    # get train test split
    train_df, test_df = get_train_test_split(df)

    # get data loaders
    train_loader = get_data_loader(
        MovieDataset, train_df, shuffle, batch_size, num_workers
    )

    test_loader = get_data_loader(
        MovieDataset, test_df, shuffle, batch_size, num_workers
    )

    # get model
    model = RecSysModel(df["userId"].max() + 1, df["movieId"].max() + 1).to(device)

    # get optimizer
    optimizer = get_optimizer(optimizer, model)

    # get loss function
    loss_function = get_loss_function(loss_function)

    # get scheduler
    scheduler = get_scheduler(optimizer, step_size, gamma)

    # train
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        loss_function,
        scheduler,
        epochs,
        device,
    )

    # test
    test(model, test_loader, loss_function, device)


if __name__ == "__main__":
    pass
