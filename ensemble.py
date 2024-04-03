from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
from item_response import irt, evaluate
import matplotlib.patches as mpatches

def resampling(data):
    N, _ = data.shape
    return data[np.random.randint(N, size=N)]

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
    lrs = [0.01, 0.005, 0.001]
    iterations = 20
    models_val_acc_lst = []
    for lr in lrs:
        resampled_data = resampling(sparse_matrix)
        theta, beta, val_acc_lst, nllk_lst = irt(resampled_data, val_data, lr, iterations)
        test_score = evaluate(data=test_data, theta=theta, beta=beta)
        models_val_acc_lst.append(val_acc_lst)
        print(f'iterations: {iterations}')
        print(f'learning rate: {lr}')
        print(f'validation_accuracies_list: {str(val_acc_lst)}')
        print(f'test accuracy: {test_score}')


    x_axis = np.arange(iterations) + 1
    names = ['lr = 0.01', 'lr = 0.005', 'lr = 0.001']
    colors = ["red", "green", "blue"]
    legends = []
    for i in range(3):
        patch = mpatches.Patch(color=colors[i], label=names[i])
        legends += [patch]


    for i in range(3):
      plt.plot(x_axis, models_val_acc_lst[i], colors[i], marker=".")

    plt.legend(handles=legends)
    plt.xlabel("iterations")
    plt.ylabel("score")
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == "__main__":
    main()
