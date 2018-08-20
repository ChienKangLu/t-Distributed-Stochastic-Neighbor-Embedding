import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    print()
    '''
        Create different distribution, then feed to tsne
    '''
    mean = (0, 0)
    cov = [[1, 0], [0, 1]]
    x = np.random.multivariate_normal(mean, cov, 50)

    mean2 = (10, 10)
    cov2 = [[1, 0], [0, 1]]
    x2 = np.random.multivariate_normal(mean2, cov2, 50)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x[:, 0], x[:, 1], marker='o', linestyle='')
    # ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
