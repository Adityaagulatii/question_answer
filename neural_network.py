from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch

def load_data(base_path="data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        inner = torch.sigmoid(torch.mm(out, torch.t(self.g.weight)) + self.g.bias)
        out = torch.sigmoid(torch.mm(inner, torch.t(self.h.weight)) + self.h.bias)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch, lb=0.0):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    epochs = []
    trains = []
    valid_accs = []


    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm() * lb/2
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t " \
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        epochs.append(epoch)
        trains.append(train_loss)
        valid_accs.append(valid_acc)
    return epochs,trains,valid_accs,valid_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    num_question = train_matrix.shape[1]
    k = [10, 50, 100, 200, 500]

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 20
    for i in k:
        model = AutoEncoder(num_question, i)
        epochs,train_losss,valid_acc,valid = train(model, lr, train_matrix, zero_train_matrix,valid_data,num_epoch)
        print("Epoch Num:",num_epoch,"Learning Rate:",lr,"K",i,"Valid Accuracy:",valid_acc[-1])

    k_sl = 100
    model_sl = AutoEncoder(num_question, k_sl)
    epoch_sl, train_loss_sl, valid_acc_sl,valid_sl = train(model_sl, lr, train_matrix, zero_train_matrix,
                                            valid_data, num_epoch)
    
    
    plt.subplot(2, 2, 1)
    plt.plot(epoch_sl, train_loss_sl)
    plt.title("Train Loss with K* = 100")
    plt.xlabel("Epoch Num")
    plt.ylabel("Train Loss")
    
    plt.subplot(2, 2, 2)
    plt.plot(epoch_sl, valid_acc_sl)
    plt.title("Valid Loss with K* = 100")
    plt.xlabel("Epoch Num")
    plt.ylabel("Valid Accuracy")
    
    plt.savefig("train_valid_loss.png")
    
    print("Valid Accuracy: ", valid_acc_sl[-1])
    print("Test Accuracy: ", evaluate(model_sl, zero_train_matrix, test_data))

    lamb = [0.001, 0.01, 0.1, 1]
    for lb in lamb:
             model = AutoEncoder(num_question, k_sl)
             epochs,train_losss,valid_acc,valid = train(model, lr, train_matrix, zero_train_matrix,valid_data,num_epoch,lb)
             print("Epoch Num:",num_epoch,"Learning Rate:",lr,"K",k_sl,"Valid Accuracy:",valid_acc[-1],"lambda",lb)
    lamb_sl = 0.001
    model_sl_lb = AutoEncoder(num_question, k_sl)
    epoch_sl_lb, train_loss_sl_lb, valid_acc_sl_lb, valid_sl_lb = train(model_sl_lb, lr, train_matrix, zero_train_matrix,
                                                            valid_data, num_epoch,lamb_sl)
    print("Valid Accuracy with L_2: ",valid_acc_sl_lb[-1])
    print("Test Accuracy with L_2: ", evaluate(model_sl_lb, zero_train_matrix, test_data))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()