from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    nbrs = KNNImputer(n_neighbors=k)
    mat = np.transpose(nbrs.fit_transform(np.transpose(matrix)))
    acc = sparse_matrix_evaluate(valid_data, mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_s =[1, 6, 11, 16, 21, 26]
    knn_user = []
    knn_item = []
    for k in k_s:
        knn = knn_impute_by_user(sparse_matrix, val_data, k)
        #print("Validation Accuracy of:",knn,"with K as",k)
        knn_user.append(knn)
    plt.plot(k_s, knn_user)
    plt.xlabel("K")
    plt.ylabel("Validation Accuracy")
    plt.xticks(k_s)
    plt.title("User validation accuracy")
    plt.savefig("User_Validation_KNN")

    for k in k_s:
        knn = knn_impute_by_item(sparse_matrix, val_data, k)
        print("Validation Accuracy of:", knn, "with K as", k)
        knn_item.append(knn)
    plt.plot(k_s, knn_item)
    plt.xlabel("K")
    plt.xticks(k_s)
    plt.ylabel("Validation Accuracy")
    plt.title("Item validation accuracy")
    plt.savefig("Item_Validation_KNN")

    print("test accuracy of k* =11 with User Validation is",knn_impute_by_user(sparse_matrix, test_data, 11))
    print("test accuracy of k* =21 with Item Validation is",knn_impute_by_item(sparse_matrix, test_data, 21))