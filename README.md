# t-Distributed-Stochastic-Neighbor-Embedding
T Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensional reduction algorithm which comes from Stochastic Neighbor Embedding (SNE). It can capture local and global structure from high dimensional data into low dimensional data

## SNE
1. Convert pairwise distances of high dimensional data into conditional probabilities(similaritiy) and assume each datapoint will pick neighbor acrroding to a Gaussain distribution,
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?p_%7Bj%5Clvert%20i%7D%3D%5Cfrac%7Bexp%28-%5Cleft%20%5C%7C%20x_i%20-%20x_j%20%5Cright%20%5C%7C%5E2/2%5Csigma%20_i%5E2%29%7D%7B%5Csum%5Cnolimits_%7Bk%5Cneq%20i%7Dexp%28-%5Cleft%20%5C%7C%20x_i%20-%20x_k%20%5Cright%20%5C%7C%5E2/2%5Csigma%20_i%5E2%29%7D" />
  </p>
  
2. Each datapoint has its own particular variance <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csigma_i" /> which can reflect how dense or sparse different region is. A variance <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csigma_i" /> can induce a probability distribution <img src="https://latex.codecogs.com/svg.latex?P_i" />. For Selecting proper variance for each <i>i</i>, user can set a fixed perplexity and it will use binary search to find <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csigma_i" /> which can let <img src="https://latex.codecogs.com/svg.latex?P_i" /> to be a distribution with the fixed perplexity,
  <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Blr%7D%20Perp%28P_i%29%3D2%5E%7BH%28P_i%29%7D%5C%5C%20H%28P_i%29%3D-%5Csum%5Cnolimits_j%20p_%7Bj%7Ci%7Dlog_2p_%7Bj%7Ci%7D%20%5Cend%7Barray%7D" />
  </p>
  
3. Covert low dimensional data into conditional probabilities(similaritiy) with the same way but set the variance to <img src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D" />,
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?q_%7Bj%5Clvert%20i%7D%3D%5Cfrac%7Bexp%28-%5Cleft%20%5C%7C%20y_i%20-%20y_j%20%5Cright%20%5C%7C%5E2%29%7D%7B%5Csum%5Cnolimits_%7Bk%5Cneq%20i%7Dexp%28-%5Cleft%20%5C%7C%20y_i%20-%20y_k%20%5Cright%20%5C%7C%5E2%29%7D" />
  </p>
  
4. Use gradient discnet to minimize Kullback-Leibler divergence(KL-divergence) of these two distribution,
  <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?C%3D%5Csum%5Cnolimits_iKL%28P_i%7C%7CQ_i%29%3D%5Csum%5Cnolimits_i%5Csum%5Cnolimits_jp_%7Bj%7Ci%7Dlog%5Cfrac%7Bp_%7Bj%7Ci%7D%7D%7Bq_%7Bj%7Ci%7D%7D" />
  </p>
  
   
  
   


## Core idea
1. Assume the original high dimensional data is created by Gaussian distribution
2. View the low dimensional data as t-distribution
## Algorithm
1. 


