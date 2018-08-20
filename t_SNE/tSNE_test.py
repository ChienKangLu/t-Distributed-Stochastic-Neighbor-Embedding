import file
from TSNE import TSNE
import numpy as np
import pandas as pd


def main():
    """
    1. random 產生 high dimension data
    2. create_dis_sigmas_prob
    3. create_p_q
    4. gradient descent(without momentum)
    """
    dir_name = "File"

    label = np.zeros(200)
    '''
    #  step 1 & step 2
    # '''
    raw_high_file_name = "high.json"
    file.create_file(raw_high_file_name, dir_name)
    rewrite = True
    if rewrite:
        raw_high_data = np.random.randint(100, size=(50, 5))
        file.write_json(file.numpy_array_to_list(raw_high_data), raw_high_file_name, dir_name)
    high_data = file.list_to_numpy_array(file.read_json(raw_high_file_name, dir_name))
    print(high_data)
    tsne = TSNE(label, high_data, dir_name)

    tsne.create_dis_sigmas_prob()

    '''
    #  step 3
    '''
    # high_dim_vectors = file.list_to_numpy_array(file.read_json("high.json", dir_name))
    # dis = file.list_to_numpy_array(file.read_json("dis.json", dir_name))
    # sigmas = file.list_to_numpy_array(file.read_json("sigmas.json", dir_name))
    # prob = file.list_to_numpy_array(file.read_json("prob.json", dir_name))
    #
    # dis_df = pd.DataFrame(dis)
    # prob_df = pd.DataFrame(prob)
    # sigmas_df = pd.DataFrame(sigmas)
    #
    # tsne = TSNE(label, high_dim_vectors,dir_name)
    # tsne.fromfile(dis,prob,sigmas)
    # tsne.create_p()

    '''
    # step 4
    '''
    # tsne.train(500, 10, momentum=0.9)  # momentum=0.9
    # print("over")


if __name__ == "__main__":
    main()
