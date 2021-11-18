import numpy as np
import random
import pickle


def load_and_get_set(test_or_train, feature_file_path, label_file_path):
    threshold = 52.3
    if test_or_train == "test":
        threshold = 48

    # loading set features
    f = open(feature_file_path, "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > threshold]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading set labels
    f = open(label_file_path, "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # preparing our training or test sets - joining datasets and lables
    myset = []

    for i in range(len(test_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(4, 1)
        myset.append((test_set_features[i].reshape(102, 1), label))

    # shuffle
    random.shuffle(myset)

    # print size
    print("The loaded set's size: ", len(myset))

    return myset


if __name__ == '__main__':
    load_and_get_set("train", "Datasets/train_set_features.pkl", "Datasets/train_set_labels.pkl")
    # ------------
    load_and_get_set("test", "Datasets/test_set_features.pkl", "Datasets/test_set_labels.pkl")
