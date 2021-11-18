import numpy as np
from ANN_Project_Assets import Loading_Datasets as ld

train_set = ld.load_and_get_set(test_or_train="train",
                                feature_file_path="ANN_Project_Assets/Datasets/train_set_features.pkl",
                                label_file_path="ANN_Project_Assets/Datasets/train_set_labels.pkl")

layer_sizes = [len(train_set[0][0]), 150, 60, 4]  # [102, 150, 60, 4]
print(layer_sizes)
print(layer_sizes[1])


