from utils import *
import matplotlib.pyplot as plt

import numpy as np

computed = None
var = None
mean = None

def _D(data, j):
    N, _ = data.shape
    global computed, var, mean

    if computed is None: 
        computed = np.nansum(data, axis=0)
        var = np.var(computed)
        mean = np.mean(computed)
    
    s = computed[j]

    # 25 is good

    return np.divide(mean, s)

def _A(data, theta, beta, i, j): 
    return (data[i][j]) *  (_D(data, j) * (theta[i] - beta[j]) )

# w.r.t theta
def _dA_theta(data, i, j): 
    return (2 * data[i][j] - 1) * _D(data, j)

# w.r.t beta
def _dA_beta(data, i, j): 
    return -(2 * data[i][j] - 1) * _D(data, j)


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for index, _ in np.ndenumerate(data):
        i, j = index
        if np.isnan(data[i][j]):
            continue
        x = _A(data, theta, beta, i, j)
        log_lklihood += x - np.log(1 + np.exp(x))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def theta_prime(data, beta, theta):
    d_theta = np.zeros(theta.shape)
    N, M = data.shape
    for i in range(N):
        for j in range(M):
            if np.isnan(data[i][j]):
                continue
            d_theta[i] += _dA_theta(data, i, j) * \
                (1 - sigmoid(_A(data, theta, beta, i, j)))
    return d_theta


def beta_prime(data, beta, theta):
    d_beta = np.zeros(beta.shape)
    N, M = data.shape
    for j in range(M):
        for i in range(N):
            if np.isnan(data[i][j]):
                continue
            d_beta[j] += _dA_beta(data, i, j) * \
                (1 - sigmoid(_A(data, theta, beta, i, j)))
    return d_beta


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_p = theta_prime(data, beta, theta)
    beta_p = beta_prime(data, beta, theta)
    theta, beta = theta + theta_p * lr, beta + beta_p * lr
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    students, questions = data.shape
    theta = np.ones(shape=(students,))
    beta = np.ones(shape=(questions,))

    val_acc_lst = []
    nllk_lst = []

    data = data.toarray()

    for _ in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        nllk_lst.append(neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, nllk_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    # train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 20
    theta, beta, val_acc_lst, nllk_lst = irt(sparse_matrix, val_data, lr, iterations)

    test_score = evaluate(data=test_data, theta=theta, beta=beta)


    print(f'iterations: {iterations}')
    print(f'learning rate: {lr}')
    print(f'validation_accuracies_list: {str(val_acc_lst)}')
    print(f'test accuracy: {test_score}')




    x_axis = np.arange(iterations) + 1
    plt.plot(x_axis, nllk_lst, label="Modified IRT model", color="green")
    plt.xlabel("iterations")
    plt.ylabel("Negative log likelihood")
    plt.legend()
    plt.show()

    plt.plot(x_axis, val_acc_lst, label="Modified IRT model", color="blue")
    plt.xlabel("iterations")
    plt.ylabel("score")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    theta.sort()
    lst = [10, 20, 30]
    colors = ['red', 'green', 'blue']
    for j in range(len(lst)):
        p = sigmoid(theta - beta[lst[j]])
        plt.plot(theta, p, color=colors[j], label=f'j= {str(lst[j])}')
    plt.ylabel("correct response probability")
    plt.xlabel("theta")
    plt.legend()
    plt.show()
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
    # data = load_train_sparse("data").toarray()
    # if computed is None: 
    #     computed = np.nansum(data, axis=0)
    #     var = np.var(computed)
    #     mean = np.mean(computed)
    #     min = np.min(computed)
    #     max = np.max(computed)

    # print(f'var: {var}')
    # print(f'mean: {mean}')
    # print(f'min: {min}')
    # print(f'max: {max}')