import numpy as np
import random
import matplotlib.pyplot as plt
from ANN_Project_Assets import Loading_Datasets as ld

train_set = ld.load_and_get_set(test_or_train="train",
                                feature_file_path="ANN_Project_Assets/Datasets/train_set_features.pkl",
                                label_file_path="ANN_Project_Assets/Datasets/train_set_labels.pkl")

layer_sizes = [len(train_set[0][0]), 150, 60, 4]  # [102, 150, 60, 4]

# There are 4 layers, between each two consecutive layers, there needs to be a weight matrix
W = [
    np.random.normal(size=(layer_sizes[1], layer_sizes[0])),  # weights between layer 0 and 1, aka W[0]
    np.random.normal(size=(layer_sizes[2], layer_sizes[1])),  # weights between layer 1 and 2, aka W[1]
    np.random.normal(size=(layer_sizes[3], layer_sizes[2]))  # weights between layer 2 and 3, aka W[2]
]

# Initialize bias to 0, for every layer.
B = [
    np.zeros((layer_sizes[1], 1)),  # bias vector between layer 0 and 1, aka B[0]
    np.zeros((layer_sizes[2], 1)),  # bias vector between layer 1 and 2, aka B[1]
    np.zeros((layer_sizes[3], 1)),  # bias vector between layer 2 and 3, aka B[2]
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def check_accuracy(calculated_labels, correct_labels):
    calculated_ans = np.where(calculated_labels == np.amax(calculated_labels))
    correct_ans = np.where(correct_labels == np.amax(correct_labels))

    return calculated_ans == correct_ans


def run_feed_forward():
    data_size = 200  # number of train set elements taken from the train set
    correct_ans_count = 0  # number of correct answers, initialized at 0

    for td in train_set[:data_size]:
        z = [
            np.zeros((layer_sizes[0], 1)),
            np.zeros((layer_sizes[1], 1)),
            np.zeros((layer_sizes[2], 1)),
            np.zeros((layer_sizes[3], 1))

        ]

        # values of the first layer (0th), initialized as the train data
        z[0] = td[0]
        np.reshape(z[0], (102, 1))

        for i in range(1, 4):
            # for each next layer, z is calculated as discussed below
            z[i] = sigmoid(W[i - 1] @ z[i - 1] + B[i - 1])

        if check_accuracy(z[3], td[1]):
            correct_ans_count += 1

    return correct_ans_count / data_size


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Hyper parameters
batch_size = 10
learning_rate = 0.6
epoch_number = 5
costs = []


# ----------------------------------------

def run_vectorized_back_propagation():
    data_size = 200
    trimmed_train_set = train_set[:data_size]

    for i in range(0, epoch_number):
        # shuffle the train set
        random.shuffle(trimmed_train_set)
        batches = [train_set[x:x + batch_size] for x in range(0, data_size, batch_size)]
        for batch in batches:

            grad_W = [
                np.random.normal(size=(layer_sizes[1], layer_sizes[0])),
                np.random.normal(size=(layer_sizes[2], layer_sizes[1])),
                np.random.normal(size=(layer_sizes[3], layer_sizes[2]))
            ]

            grad_B = [
                np.zeros((layer_sizes[1], 1)),
                np.zeros((layer_sizes[2], 1)),
                np.zeros((layer_sizes[3], 1))
            ]
            for td in batch:
                z = [
                    np.zeros((layer_sizes[0], 1)),
                    np.zeros((layer_sizes[1], 1)),
                    np.zeros((layer_sizes[2], 1)),
                    np.zeros((layer_sizes[3], 1))
                ]

                # values of the first layer (0th), initialized as the train data
                z[0] = td[0]
                np.reshape(z[0], (102, 1))

                for j in range(1, 4):
                    # for each next layer, z is calculated as discussed below
                    z[j] = sigmoid(W[j - 1] @ z[j - 1] + B[j - 1])

                # ** layer 4 to 3
                grad_B[2] += (2 * d_sigmoid(z[3]) * (z[3] - td[1]))  # bias layer 4
                grad_W[2] += (2 * d_sigmoid(z[3]) * (z[3] - td[1])) @ np.transpose(z[2])

                # delta_2 = np.zeros((layer_sizes[2], 1))
                delta_2 = (np.transpose(W[2])) @ (2 * d_sigmoid(z[3]) * (z[3] - td[1]))

                # ** layer 3 to 2
                grad_B[1] += delta_2 * d_sigmoid(z[2])  # bias layer 3
                grad_W[1] += (delta_2 * d_sigmoid(z[2])) @ np.transpose(z[1])

                # delta_1 = np.zeros((layer_sizes[1], 1))
                delta_1 = np.transpose(W[1]) @ (2 * d_sigmoid(z[2]) * delta_2)

                # ** layer 2 to 1
                grad_B[0] += delta_1 * d_sigmoid(z[1])  # bias layer 2
                grad_W[0] += delta_1 * d_sigmoid(z[1]) @ np.transpose(z[0])

            # update, using the gradient
            for ind in range(0, 3):
                W[ind] -= learning_rate * (grad_W[ind] / batch_size)
                B[ind] -= learning_rate * (grad_B[ind] / batch_size)

        cost = 0
        correct_ans_count = 0
        for td in trimmed_train_set:
            z = [
                np.zeros((layer_sizes[0], 1)),
                np.zeros((layer_sizes[1], 1)),
                np.zeros((layer_sizes[2], 1)),
                np.zeros((layer_sizes[3], 1))
            ]
            # values of the first layer (0th), initialized as the train data
            z[0] = td[0]
            np.reshape(z[0], (102, 1))

            for it in range(1, 4):
                # for each next layer, z is calculated as discussed below
                z[it] = sigmoid(W[it - 1] @ z[it - 1] + B[it - 1])

            for j in range(layer_sizes[3]):
                cost += np.power((z[3][j, 0] - td[1][j, 0]), 2)

            if check_accuracy(z[3], td[1]):
                correct_ans_count += 1

        print(correct_ans_count)
        print(data_size)
        cost /= data_size
        costs.append(cost)


run_vectorized_back_propagation()

epoch_size = [x for x in range(epoch_number)]
plt.plot(epoch_size, costs)
plt.show()
print("Back Propagation Accuracy: ", run_feed_forward())
