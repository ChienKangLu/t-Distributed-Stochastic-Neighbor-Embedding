import os
import json
import numpy as np
import gzip # for mnist
from six.moves import cPickle as pickle # for mnist


def main():
    print("我是" + __file__)
    file_name = "data.json"
    dir_name = "file"
    create_file(file_name, dir_name)
    data = [1, 2, 3, 4, 5, 6]

    write_json(data, file_name, dir_name)
    read_json(file_name, dir_name)


def write_json(data, file_name, dir_name):
    # Writing JSON data

    with open(dir_name + "\\" + file_name, 'w') as f:
        json.dump(data, f,indent=4)


def read_json(file_name, dir_name):
    # Reading data back
    data = None
    with open(dir_name + "\\" + file_name, 'r') as f:
        data = json.load(f)
    # print(data)
    return data


def list_to_numpy_array(list_data):
    npa = np.asarray(list_data, dtype=np.float32)
    return npa


def numpy_array_to_list(numpy_array):
    tolist = np.array(numpy_array).tolist()
    return tolist


def create_file(file_name, dir_name):
    path = create_dir(dir_name)
    full_file_name = path + "\\" + file_name
    if os.path.exists(full_file_name):
        print(full_file_name, "已經存在")
        rebuilt = input("是否要刪除並重新建立?")
        if rebuilt == "y":
            os.remove(full_file_name)
            open(full_file_name, "w")
            print("已經重新建立" + file_name + "檔案" + "(" + os.path.realpath(full_file_name) + ")")
        else:
            print("使用舊的" + file_name + "檔案" + "(" + os.path.realpath(full_file_name) + ")")
    else:
        print("已經建立" + file_name + "資料夾" + "(" + os.path.realpath(full_file_name) + ")")
        open(full_file_name, "w")


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("建立" + dir_name + "資料夾")
    else:
        print("已經建立" + dir_name + "資料夾" + "(" + os.path.realpath(dir_name) + ")")
    return os.path.realpath(dir_name)


def load_mnist(digits_to_keep=[0, 1 , 2, 8, 9], n=200):
    path = os.path.realpath("datasets")
    print(path)
    # Load the dataset
    f = gzip.open(path+"\\"+"mnist.pkl.gz", 'rb')
    train_set, _, _ = pickle.load(f, encoding='latin1')
    f.close()

    # Find indices of digits in training set that we will keep
    includes_matrix = [(train_set[1] == i) for i in digits_to_keep]
    keep_indices = np.sum(includes_matrix, 0).astype(np.bool)

    # Drop images of any other digits
    train_set = [train_set[0][keep_indices], train_set[1][keep_indices]]

    # Only keep the first N examples
    n = min(n, train_set[0].shape[0])
    train_set = [train_set[0][:n], train_set[1][:n]]

    return train_set

if __name__ == "__main__":
    main()
