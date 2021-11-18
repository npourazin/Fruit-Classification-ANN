import numpy as np
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


print("Feed Forward Accuracy: ", run_feed_forward())
