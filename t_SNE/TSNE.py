import numpy as np
import math
import pandas as pd
import file
import matplotlib.pyplot as plt


def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def binary_search(index, eval_fn, target, threshold=1e-10, max_iter=1000, lower=1e-20, upper=1000.):
    guess = 0
    for i in range(max_iter):
        guess = (lower + upper) / 2
        value = eval_fn(guess, index)
        if value > target:
            upper = guess
        else:
            lower = guess
        if np.abs(value - target) <= threshold:
            break
    return guess


def shannon_entropy(row_probs):
    # row_probs = probs[i:i + 1, :]  # get ith row with all column
    n = len(row_probs)
    log_sum = 0
    for j in range(n):
        log_sum += row_probs[j] * np.log2(row_probs[j])
    entropy = -log_sum
    return entropy


def perplexity(row_probs):
    perp = np.power(2, shannon_entropy(row_probs))
    return perp


def scatter_2d(point2d, class_indexes, n_iter, dir_name, ms=3, alpha=0.1, momentum=None, save=None):
    """

    :param save:
    :param momentum:
    :param dir_name:
    :param n_iter:
    :param point2d:
    :param class_indexes:
    :param ms: marker size
    :param alpha:
    :param savename:
    :return:
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    classes = list(np.unique(class_indexes))
    markers = 'os' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(point2d[class_indexes == cls, 0], point2d[class_indexes == cls, 1], marker=mark,
                linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
                markeredgecolor='black', markeredgewidth=0.4)
    ax.legend()
    # plt.show()

    if save:
        '''
        # save picture at every iteration
        '''
        pic_name = "iter"
        if momentum:
            pic_name = "momentum_" + pic_name
        file.create_dir(dir_name + "\\" + "pic")
        plt.savefig(dir_name + "\\" + "pic\\" + pic_name + str(n_iter))
        '''
        # save low dimension vectors at every iteration
        '''
        low_file_name = "low_" + str(n_iter) + ".json"
        if momentum:
            low_file_name = "momentum_" + low_file_name
        dir_name = dir_name
        file.create_file(low_file_name, dir_name)
        file.write_json(file.numpy_array_to_list(point2d), low_file_name, dir_name)

    return ax


class TSNE:
    def __init__(self, labels, high_dim_vectors, dir_name):
        """
        :param high_dim_vectors: npArray
        """
        self.high_dim_vectors = high_dim_vectors
        self.n = len(self.high_dim_vectors)
        self.dis = np.zeros((self.n, self.n))
        self.prob = np.zeros((self.n, self.n))
        self.sigmas = np.zeros(self.n)

        self.low_dim_vectors = np.random.RandomState(1).normal(0., 0.0001, [high_dim_vectors.shape[0], 2])
        self.low_dis = np.zeros((self.n, self.n))
        self.low_prob = np.zeros((self.n, self.n))
        self.p = None
        self.q = None
        self.dir_name = dir_name
        self.labels = labels

    def fromfile(self, dis, prob, sigmas):
        self.dis = dis
        self.prob = prob
        self.sigmas = sigmas

    def probability(self, i, j, sigma_i):
        """
        將歐幾里得距離轉換為條件概率來表達點與點之間的相似度(Pj|i)

        numerator分子，denominator分母
        :param i: center at Xi
        :param j: Xj
        :param sigma_i:
        :return: probability
        """
        numer = math.exp(- np.power(self.dis[i][j], 2) / (2 * (sigma_i ** 2)))
        denom = 0
        for k, vector in enumerate(self.high_dim_vectors):
            if k != i:
                denom += math.exp(- np.power(self.dis[i][k], 2) / (2 * (sigma_i ** 2)))
        return numer / denom

    def calculate_dis(self):
        for i in range(self.n):
            for j in range(self.n):
                self.dis[i][j] = distance(self.high_dim_vectors[i], self.high_dim_vectors[j])

    def calculate_row_prob(self, i):
        for j in range(self.n):
            if j != i:
                self.prob[i][j] = self.probability(i, j, self.sigmas[i])
        self.prob[i][i] = 1e-10  # set to 0, for preventing np.log2(0), set 0 as a very small float

    def eval_fn(self, sigma, i):
        # print(sigma)
        self.sigmas[i] = sigma
        self.calculate_row_prob(i)
        return perplexity(self.prob[i])

    def find_sigmas(self):
        for i in range(self.n):
            correct_sigma = binary_search(i, self.eval_fn, 20)
            self.sigmas[i] = correct_sigma
            print(i, self.sigmas[i])

    def create_dis_sigmas_prob(self):
        self.calculate_dis()
        self.find_sigmas()
        for i in range(self.n):
            self.calculate_row_prob(i)

        dis_df = pd.DataFrame(self.dis)
        prob_df = pd.DataFrame(self.prob)
        sigmas_df = pd.DataFrame(self.sigmas)

        prob_file_name = "prob.json"
        dir_name = self.dir_name
        file.create_file(prob_file_name, dir_name)
        file.write_json(file.numpy_array_to_list(self.prob), prob_file_name, dir_name)

        dis_file_name = "dis.json"
        dir_name = self.dir_name
        file.create_file(dis_file_name, dir_name)
        file.write_json(file.numpy_array_to_list(self.dis), dis_file_name, dir_name)

        sigmas_file_name = "sigmas.json"
        dir_name = self.dir_name
        file.create_file(sigmas_file_name, dir_name)
        file.write_json(file.numpy_array_to_list(self.sigmas), sigmas_file_name, dir_name)

    def create_p(self):
        self.p = self.p_joint()
        p_df = pd.DataFrame(self.p)
        print()

    def create_q_low_distance(self):
        self.calculate_low_dis()
        self.q = self.q_joint()
        low_dis_df = pd.DataFrame(self.low_dis)
        q_df = pd.DataFrame(self.q)

    # def create_p_q(self):
    #     self.calculate_low_dis()
    #     self.q = self.q_joint()
    #     self.p = self.p_joint()
    #     low_dis_df = pd.DataFrame(self.low_dis)
    #     q_df = pd.DataFrame(self.q)
    #     p_df = pd.DataFrame(self.p)
    #     print()

    def p_joint(self):
        """(Pi|j+Pj|i)/(2*N)"""
        return (self.prob + self.prob.T) / (2. * self.n)

    def q_joint(self):
        q = np.zeros((self.n, self.n))
        sum_dis = 0
        for k in range(self.n):
            for l in range(self.n):
                if k != l:
                    sum_dis += math.pow(1 + np.power(self.low_dis[k][l], 2),
                                        -1)  # sum_dis += math.exp(- self.low_dis[k][l])

        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    q[i][j] = self.probability_without_sigma(i, j, sum_dis)
        return q

    def probability_without_sigma(self, i, j, sum_dis):
        numer = math.pow(1 + np.power(self.low_dis[i][j], 2), -1)  # numer = math.exp(- self.low_dis[i][j])
        denom = sum_dis
        return numer / denom

    def calculate_low_dis(self):
        for i in range(self.n):
            for j in range(self.n):
                self.low_dis[i][j] = distance(self.low_dim_vectors[i], self.low_dim_vectors[j])

    def gradient(self, i):
        terms = 0
        for j in range(self.n):
            terms += (self.p[i][j] - self.q[i][j]) * (self.low_dim_vectors[i] - self.low_dim_vectors[j]) * np.power(
                1 + distance(self.low_dim_vectors[i], self.low_dim_vectors[j]), -1)
        return 4 * terms

    def kl_divergence(self):
        kl = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    kl += self.p[i][j] * np.log2(self.p[i][j] / self.q[i][j])
        return kl

    def train(self, num_iters, learning_rate, momentum=None):
        """gradient descent"""

        # momentum
        if momentum:
            low_dim_vectors_m2 = self.low_dim_vectors.copy()
            low_dim_vectors_m1 = self.low_dim_vectors.copy()

        for n_iter in range(num_iters):
            self.create_q_low_distance()
            grads = np.zeros((self.n, 2))
            for i in range(self.n):
                grads[i] = self.gradient(i)
            self.low_dim_vectors = self.low_dim_vectors - learning_rate * grads

            # momentum (need to consider the same direction!)
            if momentum:
                for i in range(self.n):
                    if np.inner((low_dim_vectors_m1[i] - low_dim_vectors_m2[i]), grads[i]) > 0:
                        self.low_dim_vectors[i] += momentum * (low_dim_vectors_m1[i] - low_dim_vectors_m2[i])
                # momentum Update previous low_dim_vectors's for momentum
                low_dim_vectors_m2 = low_dim_vectors_m1.copy()
                low_dim_vectors_m1 = self.low_dim_vectors.copy()

            # hope KL divergence small, want 2 distribution are similar
            # For debug purpose, we need to print the cost to make sure the gradient is converged!!!!(important)
            print(n_iter, "C=", self.kl_divergence())

            if n_iter == num_iters - 1 or n_iter % (num_iters / 5) == 0:
                print("print")
                scatter_2d(self.low_dim_vectors, self.labels, n_iter, self.dir_name, alpha=1.0, ms=10,
                           momentum=momentum, save=True)
