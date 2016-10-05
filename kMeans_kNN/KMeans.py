import numpy as np
import pandas as pd
import time


class KMeans(object):
    """K-Means clustering algorithm"""
    def __init__(self, k):
        self.k = k

    def cluster(self, X):
        """Perform KMeans clustering to find the centers of clusters"""
        # Initialization
        random_indexes = np.random.choice(len(X), self.k)
        centers = np.array([X[ind] for ind in random_indexes])
        clusters = [0] * (len(X))

        conti = False
        while True:
            # Find optimal partition of the data points given the centers
            for ix, x in enumerate(X):
                best = float("inf")

                for ind, center in enumerate(centers):
                    dist = np.sum((x - center)**2)
                    if dist < best:
                        best = dist
                        clus = ind

                if clus != clusters[ix]:
                    clusters[ix] = clus
                    conti = True

            # Re-compute the optimal centers given the partition
            add = {}
            for k in range(self.k):
                add[k] = np.zeros(len(X[0]))

            num = [0] * self.k
            for ix, ind in enumerate(clusters):
                add[ind] += X[ix]
                num[ind] += 1

            centers = np.array([add[j] / num[j] for j in range(self.k)])
            # stop if the partition hasn't changed
            if conti:
                conti = False
            else:
                break

        self.centers = centers
        self.clusters = clusters
        self.data = X

        return centers, clusters

    def square_error(self, Print=False):
        """Compute average square error to centers"""
        Error = 0
        for i in range(len(self.data)):
            Error += np.sum((self.data[i] - self.centers[self.clusters[i]])**2)

        Error = Error / len(self.data)
        if Print:
            print("Average square error to centers: %.4f" % Error)

        return Error


def run_KMeans(base_dir='./'):
    Data = pd.read_csv(base_dir + 'Data/hw4_kmeans_train.dat',
                       header=None, delim_whitespace=True)
    print('Data loaded\n', Data.head())

    print('\nStart clustering...')
    start = time.clock()
    Ein = []

    for i in range(500):
        K = KMeans(10)
        centers, clusters = K.cluster(Data.values)
        Ein.append(K.square_error())

    print('\tAveraged Ein: %.4f, using %.2f seconds.' %
          (np.mean(Ein), time.clock() - start))
    print('\nCenters:\n', centers)


def main():
    run_KMeans()


if __name__ == '__main__':
    main()
