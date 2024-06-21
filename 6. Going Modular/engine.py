import torch
from torch import nn

def accuracy_fn(y_pred, y_true):
    """
    Computes the accuracy of predictions.

    Args:
        y_pred (torch.Tensor): The predicted labels.
        y_true (torch.Tensor): The true labels.

    Returns:
        float: The accuracy percentage of the predictions.
    """
    correct = torch.eq(y_pred, y_true).sum().item()
    return (correct/len(y_true)) * 100


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    """
    Performs a single training step for the given model.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the training data.
        loss_fn (nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters.
        device (torch.device): The device to run the training on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average training loss and training accuracy.
    """


    # put the model on train mode
    model.train()
    
    train_loss, train_acc = 0, 0

    for X_train, y_train in dataloader:
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        # forward pass
        y_logits = model(X_train)
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

        # calc loss, acc
        loss = loss_fn(y_logits, y_train)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_pred=y_pred, y_true=y_train)

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backwards (back_propagation)
        loss.backward()

        # optimizer step (update parameters)
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device):
    """
    Performs a single evaluation step for the given model.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the test data.
        loss_fn (nn.Module): The loss function to use.
        device (torch.device): The device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average test loss and test accuracy.
    """
    test_loss, test_acc = 0, 0

    # put the model in eval() model
    model.eval()

    with torch.inference_mode():
        for X_test, y_test in dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            # forward pass
            y_logits = model(X_test)
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

            # calc loss, acc
            test_loss += loss_fn(y_logits, y_test).item()
            test_acc += accuracy_fn(y_pred=y_pred, y_true=y_test)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          device,
          epochs: int = 5):
    """
    Trains the given model using the specified data loaders, loss function, optimizer, and device.

    Args:
        model (nn.Module): The neural network model to train.
        train_dataloader (torch.utils.data.DataLoader): The DataLoader providing the training data.
        test_dataloader (torch.utils.data.DataLoader): The DataLoader providing the test data.
        loss_fn (nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters.
        device (torch.device): The device to run the training on (e.g., 'cpu' or 'cuda').
        epochs (int, optional): The number of epochs to train the model (default is 5).

    Returns:
        dict: A dictionary containing the training and testing results including losses and accuracies.
    """
    
    # send the model to target device
    model = model.to(device)

    # creating a empty result dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch:{epoch}\tTrain_Loss:{train_loss:.4f}\tTrain_Acc:{train_acc:.4f}\tTest_Loss:{test_loss:.4f}\tTest_Acc:{test_acc:.4f}")

        # update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
            