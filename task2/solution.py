import numpy as np
import torch
import os
import sys
import math
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm
from utils import *
from modules import *


def train_network(model, optimizer, train_loader, num_epochs=100, pbar_update_interval=100):
    '''
    Updates the model parameters (in place) using the given optimizer object.
    Returns `None`.

    The progress bar computes the accuracy every `pbar_update_interval`
    iterations.
    '''
    criterion = torch.nn.CrossEntropyLoss() # always used in this assignment

    pbar = trange(num_epochs)
    for i in pbar:
        for k, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.squeeze()
            model.zero_grad()
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            if type(model) == BayesNet:
                # BayesNet implies additional KL-loss.
                # TODO: enter your code here
                kl_weight = 1
                kl_loss = model.kl_loss()
                loss = loss + kl_weight * kl_loss

            loss.backward()
            optimizer.step()

            if k % pbar_update_interval == 0:
                acc = (model(batch_x).argmax(axis=1) == batch_y).sum().float()/(len(batch_y))
                pbar.set_postfix(loss=loss.item(), acc=acc.item())


def loadBayesNet(file_path, input_size, num_layers, width):
    model = BayesNet(input_size, num_layers, width)
    model.load_state_dict(torch.load(file_path))
    return model


def main(test_loader=None, private_test=False):
    batch_size = 128  # Try playing around with this
    model_type = "bayesnet"  # Try changing this to "densenet" as a comparison
    extended_evaluation = False  # Set this to True for additional model evaluation

    dataset_train = load_rotated_mnist()
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, drop_last=True)

    if model_type == "bayesnet":
        model = BayesNet(input_size=784, num_layers=2, width=50)
    elif model_type == "densenet":
        model = Densenet(input_size=784, num_layers=2, width=100)

    model = loadBayesNet("bn_5x100_m0s015.pt", 28**2, 5, 100)

    if test_loader is None:
        print("evaluating on train data")
        test_loader = train_loader
    else:
        print("evaluating on test data")

    # Do not change this! The main() method should return the predictions for the test loader
    predictions = evaluate_model(model, model_type, test_loader, batch_size, extended_evaluation, private_test)
    return predictions


if __name__=="__main__":
    main()
