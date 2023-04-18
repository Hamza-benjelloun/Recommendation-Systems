from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch


def encode_data(df):
    return LabelEncoder().fit_transform(df)


def get_train_test_split(df, test_size=0.1, random_state=42, stratify=None):
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )


def get_data_loader(dataset, df, shuffle, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset(
            users=df["userId"].values,
            movies=df["movieId"].values,
            ratings=df["rating"].values,
        ),
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_optimizer(optimizer, model):
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.001)
    else:
        raise ValueError("Optimizer not supported")


def get_loss_function(loss_function):
    if loss_function == "mse":
        return torch.nn.MSELoss()
    elif loss_function == "bce":
        return torch.nn.BCELoss()
    else:
        raise ValueError("Loss function not supported")


def get_scheduler(optimizer, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_function,
    scheduler,
    epochs,
    device,
):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)

            optimizer.zero_grad()

            outputs = model(users, movies)
            loss = loss_function(outputs, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                users = batch["users"].to(device)
                movies = batch["movies"].to(device)
                ratings = batch["ratings"].to(device)

                outputs = model(users, movies)
                loss = loss_function(outputs, ratings)

                test_loss += loss.item()

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss/len(train_loader):.3f}, Test Loss: {test_loss/len(test_loader):.3f}"
        )


def test(model, test_loader, loss_function, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            users = batch["users"].to(device)
            movies = batch["movies"].to(device)
            ratings = batch["ratings"].to(device)

            outputs = model(users, movies)
            loss = loss_function(outputs, ratings)

            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader):.3f}")
