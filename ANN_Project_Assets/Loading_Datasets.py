import numpy as np
import random
import pickle


def load_and_get_train_set(feature_file_path, label_file_path):
    # loading training set features
    f = open(feature_file_path, "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 52.3]

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open(label_file_path, "rb")
    train_set_labels = pickle.load(f)
    f.close()

    # preparing our training and test sets - joining datasets and lables
    train_set = []

    for i in range(len(train_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(train_set_labels[i])] = 1
        label = label.reshape(4, 1)
        train_set.append((train_set_features[i].reshape(102, 1), label))

    # shuffle
    random.shuffle(train_set)

    # print size
    print(len(train_set))  # 1962

    return train_set


def load_and_get_test_set(feature_file_path, label_file_path):
    # loading test set features
    f = open(feature_file_path, "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 48]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open(label_file_path, "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # preparing our training and test sets - joining datasets and lables
    test_set = []

    for i in range(len(test_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(4, 1)
        test_set.append((test_set_features[i].reshape(102, 1), label))

    # shuffle
    random.shuffle(test_set)

    # print size
    print(len(test_set))  # 662

    return test_set


if __name__ == '__main__':
    load_and_get_train_set("Datasets/train_set_features.pkl", "Datasets/train_set_labels.pkl")
    # ------------
    load_and_get_test_set("Datasets/test_set_features.pkl", "Datasets/test_set_labels.pkl")
