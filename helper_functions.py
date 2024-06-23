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


def plot_loss_curve(results: Dict[str, List[float]]):
    train_loss  = results["train_loss"]
    train_acc = results["train_acc"]
    test_loss = results["test_loss"]
    test_acc = results["test_acc"]

    epochs = range(len(results['train_loss']))

    # setting up the plot
    plt.figure(figsize=(15, 7))

    # plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    # plot the acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
