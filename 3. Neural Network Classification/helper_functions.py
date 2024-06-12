import os
import torch
import requests
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10,7))
    
    #plot training data
    plt.scatter(train_data, train_labels, c='b', s=4, label='training data')
    
    #plot test data
    plt.scatter(test_data, test_labels, c='g', s=4, label='testing data')
    
    # if predictions then plot 
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='predictions')
        
    # show the legend
    plt.legend(prop={'size':14})


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots Decision boundaries of model predicting on X in comparison to y."""

    # putting everything to cpu for numpy and matplotlib
    model.to("cpu")
    X, y, = X.to("cpu"), y.to("cpu")

    # set up prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # test for multi class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # multi class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary

    # reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contour(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
