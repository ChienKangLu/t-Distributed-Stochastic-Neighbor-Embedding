# t-Distributed-Stochastic-Neighbor-Embedding
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensional reduction algorithm which comes from Stochastic Neighbor Embedding (SNE). It can capture local and global structure from high dimensional data into low dimensional data

## SNE
1. Convert pairwise distances of high dimensional data into conditional probabilities(similaritiy) and assume each datapoint will pick neighbor according to a Gaussain distribution,
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?p_%7Bj%5Clvert%20i%7D%3D%5Cfrac%7Bexp%28-%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20-%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D%20%5Cright%20%5C%7C%5E2/2%5Csigma%20_i%5E2%29%7D%7B%5Csum%5Cnolimits_%7Bk%5Cneq%20i%7Dexp%28-%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20-%20%5Ctextbf%7B%5Ctextit%7Bx%7D%7D_%5Ctextbf%7B%5Ctextit%7Bk%7D%7D%20%5Cright%20%5C%7C%5E2/2%5Csigma%20_i%5E2%29%7D" />
  </p>             
  
2. Each datapoint of high dimensional data has its own particular variance <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csigma_i" /> which can reflect how dense or sparse different region is. A variance <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csigma_i" /> can induce a probability distribution <img src="https://latex.codecogs.com/svg.latex?P_i" />. For Selecting proper variance for each <i>i</i>, user can set a fixed perplexity and it will use binary search to find <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csigma_i" /> which can let <img src="https://latex.codecogs.com/svg.latex?P_i" /> to be a distribution with the fixed perplexity,
  <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Blr%7D%20Perp%28P_i%29%3D2%5E%7BH%28P_i%29%7D%5C%5C%20H%28P_i%29%3D-%5Csum%5Cnolimits_j%20p_%7Bj%7Ci%7Dlog_2p_%7Bj%7Ci%7D%20%5Cend%7Barray%7D" />
  </p>
  
3. Covert low dimensional data into conditional probabilities(similaritiy) with the same way but set the variance to <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D" />,
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?q_%7Bj%5Clvert%20i%7D%3D%5Cfrac%7Bexp%28-%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20-%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D%20%5Cright%20%5C%7C%5E2%29%7D%7B%5Csum%5Cnolimits_%7Bk%5Cneq%20i%7Dexp%28-%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20-%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bk%7D%7D%20%5Cright%20%5C%7C%5E2%29%7D" />
  </p>
  
4. Use gradient discnet to minimize Kullback-Leibler divergence(KL-divergence) of these two distribution,
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?C%3D%5Csum%5Cnolimits_iKL%28P_i%7C%7CQ_i%29%3D%5Csum%5Cnolimits_i%5Csum%5Cnolimits_jp_%7Bj%7Ci%7Dlog%5Cfrac%7Bp_%7Bj%7Ci%7D%7D%7Bq_%7Bj%7Ci%7D%7D" />
  </p>
  
## t-SNE
t-SNE use **symmetrized cost function** of SNE and use **Student-t distribution** to compute similarity of low dimensional data.
1. **Symmetrized cost function**
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?p_%7Bij%7D%3D%5Cfrac%7Bp_%7Bj%7Ci%7D&plus;p_%7Bi%7Cj%7D%7D%7B2n%7D" />
  </p>
  
2. **Student-t distribution**
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?q_%7Bij%7D%3D%5Cfrac%7B%281&plus;%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%20-%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D%20%5Cright%20%5C%7C%5E2%29%5E%7B-1%7D%7D%7B%5Csum%5Cnolimits_%7Bk%5Cneq%20l%7D%281&plus;%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bk%7D%7D-%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bl%7D%7D%20%5Cright%20%5C%7C%5E2%29%5E%7B-1%7D%7D" />
  </p>
  
3. KL-divergence
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?C%3D%5Csum%5Cnolimits_iKL%28P%7C%7CQ%29%3D%5Csum%5Cnolimits_i%5Csum%5Cnolimits_jp_%7Bij%7Dlog%5Cfrac%7Bp_%7Bij%7D%7D%7Bq_%7Bij%7D%7D" />
  </p>

4. Gradient
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cdelta%20C%7D%7B%5Cdelta%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D%7D%3D4%5Csum%5Cnolimits_j%28p_i_j-q_i_j%29%28%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D-%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D%29%281&plus;%5Cleft%20%5C%7C%20%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bi%7D%7D-%5Ctextbf%7B%5Ctextit%7By%7D%7D_%5Ctextbf%7B%5Ctextit%7Bj%7D%7D%20%5Cright%20%5C%7C%5E2%29%5E%7B-1%7D" />
  </p>

### Practice
1. Train with momentum: 0.9
2. Learing rate: 15
3. Iteration: 500
4. Data: MNIST
5. Result
    <table>
      <tr align="center">
        <td><img src="https://github.com/ChienKangLu/t-Distributed-Stochastic-Neighbor-Embedding/blob/master/t_SNE/File_mnist_01289_200/pic/momentum_iter0.png" /></td>
        <td><img src="https://github.com/ChienKangLu/t-Distributed-Stochastic-Neighbor-Embedding/blob/master/t_SNE/File_mnist_01289_200/pic/momentum_iter100.png" /></td>
        <td><img src="https://github.com/ChienKangLu/t-Distributed-Stochastic-Neighbor-Embedding/blob/master/t_SNE/File_mnist_01289_200/pic/momentum_iter200.png" /></td>
      </tr>
      <tr align="center">
        <td>iter 0</td>
        <td>iter 100</td>
        <td>iter 200</td>
      </tr>
      <tr align="center">
        <td><img src="https://github.com/ChienKangLu/t-Distributed-Stochastic-Neighbor-Embedding/blob/master/t_SNE/File_mnist_01289_200/pic/momentum_iter300.png" /></td>
        <td><img src="https://github.com/ChienKangLu/t-Distributed-Stochastic-Neighbor-Embedding/blob/master/t_SNE/File_mnist_01289_200/pic/momentum_iter400.png" /></td>
        <td><img src="https://github.com/ChienKangLu/t-Distributed-Stochastic-Neighbor-Embedding/blob/master/t_SNE/File_mnist_01289_200/pic/momentum_iter499.png" /></td>
      </tr>
      <tr align="center">
        <td>iter 300</td>
        <td>iter 400</td>
        <td>iter 499</td>
      </tr>
    </table>

## Reference

[Visualizing Data using t-SNE](https://www.semanticscholar.org/paper/Visualizing-Data-using-t-SNE-Maaten-Hinton/10eb7bfa7687f498268bdf74b2f60020a151bdc6)
