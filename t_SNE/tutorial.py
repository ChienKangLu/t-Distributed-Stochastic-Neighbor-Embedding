import numpy as np

import pandas as pd

def main():
    print()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])

    v = np.array([9, 10])
    w = np.array([11, 12])

    # Inner product of vectors; both produce 219
    print(v.dot(w))
    print(np.dot(v, w))

    # Matrix / vector product; both produce the rank 1 array [29 67]
    print(x.dot(v))
    print(np.dot(x, v))

    # Matrix / matrix product; both produce the rank 2 array
    # [[19 22]
    #  [43 50]]
    print(x.dot(y))
    print(np.dot(x, y))

    rVector = np.random.rand(100, 3)
    print(rVector[0])

    t = np.array([[1, 2], [3, 4]])
    print(t)  # Prints "[[1 2]
    #          [3 4]]"
    print(t.T)  # Prints "[[1 3]
    #          [2 4]]"

    # Note that taking the transpose of a rank 1 array does nothing:
    c = np.array([1, 2, 3])
    print(c)  # Prints "[1 2 3]"
    print(c.T)  # Prints "[1 2 3]"

    # loop numpy array be index
    w = np.array([[5, 6], [7, 8], [1, 3]])
    w.ndim
    N = len(w)
    print(N, w.shape[1])
    i = 2
    print(w[i:i + 1, :], w[i:i + 1, :])
    # for (i,j),value in np.ndenumerate(w):
    #     print(i,"*",j,"*",value)
    for i in range(N):
        for j in range(N):
            print(i, ":", w[i], j, ":", w[j])

    # loop list
    # list1=[[1,2],[4,5],[4,3]]
    # for i in range(len(list1)):
    #     for j in range(len(list1)):
    #         print(i,j)
    # print(np.log2(0))

    q = np.zeros((3+1, 4))
    # q[-1]=np.zeros(4)
    q_dataframe=pd.DataFrame(q)
    print()

    w = np.array([[5, 6], [7, 8]])
    z = np.array([2, 2])
    s = np.array([[2, 7],[2, 2],[2, 2]])
    qq = s[1]
    wz=w.dot(z)
    wz_dataframe=pd.DataFrame(wz)
    print()

    s_dataframe = pd.DataFrame(s[0])
    s[0]=np.array([3,8])
    yy_dataframe = pd.DataFrame(np.array([3,8]))

    for t in np.arange(4):
        print(t)
if __name__ == '__main__':
    main()
