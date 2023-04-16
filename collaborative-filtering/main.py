import torch

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


if __name__ == "__main__":
    pass
