from dataclasses import dataclass
from typing import final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset


# 0: n -> 6n
# 1: 6 * (5 - 0) / 5 = 6
# 2: 6 * (5 - 1) / 5 = 24 / 5
# 3: 6 * (5 - 2) / 5 = 18 / 5
# 4: 6 * (5 - 3) / 5 = 12 / 5
# 5: 6 * (5 - 4) / 5 = 6 / 5
def create_model(
    input_dim: int = 80,
    multiplier_each: int = 6,
    layers_each: int = 5,
):
    layers: list[nn.Module] = [
        nn.Linear(
            input_dim,
            input_dim * multiplier_each,
            dtype=torch.float64,
        ),
        nn.ELU(),
    ]

    prev_features = input_dim * multiplier_each
    for i in range(layers_each):
        out_features = input_dim * (multiplier_each * (layers_each - i) // layers_each)
        layers.append(nn.Linear(prev_features, out_features, dtype=torch.float64))
        layers.append(nn.ELU())
        prev_features = out_features

    layers.append(nn.Linear(prev_features, 2, dtype=torch.float64))
    # layers.append(nn.Softmax(dim=1))

    model = nn.Sequential(*layers)

    return model


@dataclass
class History:
    history: dict[str, list[float]]


@final
class AdmissionModel:
    def __init__(
        self,
        input_dim: int,
        learning_rate: float,
        multiplier_each: int = 6,
        layers_each: int = 5,
    ):
        self.model = create_model(input_dim, multiplier_each, layers_each)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        epochs: int,
        batch_size: int,
        shuffle: bool,
        verbose: int,
    ):
        dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        correct = 0
        losses = []
        self.model.train()
        for data, target in tqdm.tqdm(loader, "Training"):
            self.optimizer.zero_grad()
            output = self.model(data)
            # print("data", data)
            # print("target", target)
            # new_target = torch.argmax(target, dim=1)
            # loss = F.nll_loss(output, new_target)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            actual = target.data.max(1, keepdim=True)[1]
            correct += pred.eq(actual).sum().item()

            losses.append(loss.item())

        return History(
            {
                "acc": [correct / len(loader.dataset)],
                "loss": [sum(losses) / len(losses)],
            }
        )

    def predict(
        self,
        X: npt.NDArray[np.float64],
        batch_size: int,
        verbose: int,
    ):
        self.model.eval()
        data = torch.tensor(X)
        with torch.no_grad():
            return F.softmax(self.model(data), dim=1).numpy()
            # return self.model(data).numpy()

    def load_weights(self, path: str):
        state_dict = torch.load(path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
