import time
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances,manhattan_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#CLASS START=====================================================================================================================
class kmeans:
    
    def __init__(self,k):
        self.k = k
        
    #Function to read and preproccess the data
    def read_data(self):
        MNIST_df = pd.read_csv("image_new_test_MNIST.txt", header=None)
        MNIST_array = np.array(MNIST_df)
        MNIST_array = MNIST_array.astype(float)
        
        #normalization of data using minmax scaler
        scaler = MinMaxScaler()
        scaled_MNIST_array = scaler.fit_transform(MNIST_array)
        
        #dimension reduction
        pca = PCA(n_components= 30)
        pca_MNIST_array = pca.fit_transform(scaled_MNIST_array)
        
        #high dimension reduction using TSNE
        tsne = TSNE(n_components = 2, perplexity = 40, init = 'pca', random_state=0)        
        tsne_MNIST_array = tsne.fit_transform(pca_MNIST_array)
        
        return tsne_MNIST_array, MNIST_df
    
    #Function to calculate the manhattan distance
    def clustering_manhattan_distance(self, MNIST_array, centroids):    
        distance_matrix = manhattan_distances(MNIST_array, centroids)
        closest_centroids = []
        for i in range(distance_matrix.shape[0]):
            c = np.argmin(distance_matrix[i])
            closest_centroids.append(c)
        return closest_centroids

    #Function to calculate the similarity
    def clustering_cosine_similarity(self, MNIST_array, centroids):    
        distance_matrix = cosine_similarity(MNIST_array, centroids)
        closest_centroids = []
        for i in range(distance_matrix.shape[0]):
            c = np.argmax(distance_matrix[i])
            closest_centroids.append(c)
        return closest_centroids
    
    #Function to calculate euclidean distance    
    def clustering_euclidean_distance(self, MNIST_array, centroids):    
        distance_matrix = euclidean_distances(MNIST_array, centroids)
        closest_centroids = []
        for i in range(distance_matrix.shape[0]):
            c = np.argmin(distance_matrix[i])
            closest_centroids.append(c)
        return closest_centroids
    
    #Function to clculate the centroids
    def calculate_centroids(self, MNIST_array, nearest_centroid, centroids):
        cluster_d = list()
        #all_cluster_distances = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        all_cluster_distances = np.zeros(len(centroids))
        new_centroids = list()
        new_df = pd.concat([pd.DataFrame(MNIST_array), pd.DataFrame(nearest_centroid, columns=['Cluster'])], axis=1)    
        new_df_arr = np.array(new_df['Cluster'])
        for c in set(new_df_arr):        
            thiscluster = new_df[new_df['Cluster'] == c][new_df.columns[:-1]]
            temp = np.array(centroids[c])
            temp = temp.reshape(1,-1)
            #cluster_d = euclidean_distances(thiscluster, temp)
            cluster_d = manhattan_distances(thiscluster, temp)
            for d in cluster_d:
                all_cluster_distances[c] += d*d        
            cluster_mean = thiscluster.mean(axis=0)
            new_centroids.append(cluster_mean)
        return new_centroids, all_cluster_distances
    
    #Function to visualize the SSE and no.of iterations
    def visualize_sse(self, iterations, SSE):
        plt.figure()
        plt.plot(range(iterations), SSE, 'rx-')
        plt.xlabel('No.of iterations')
        plt.ylabel('SSE(Sum of squared errors)')
        plt.title('Elbow Method showing the optimal iterations')
        plt.show()        
    
    #Function to visualize the SSE and different k-values:
    def visualize_k_sse(self):
        MNIST_array, MNIST_df = self.read_data()         

        all_SSE = []
        all_k = []
        for k in range(2,21,2):
            #Randomly select three points as centroids
            centroid_index = random.sample(range(0, len(MNIST_df)), k)    
            centroids = list()
            for i in centroid_index:
                centroids.append(MNIST_array[i])
    
            #converting list into numpy array
            centroids = np.array(centroids)    
    
            #List for sum of squared errors
            SSE = list()            
            no_of_iterations = 50
            closest_centroid = list()
            for i in range(no_of_iterations):
                closest_centroid = self.clustering_manhattan_distance(MNIST_array, centroids)
                #closest_centroid = clustering_cosine_similarity(iris_array, centroids)
                centroids, all_cluster_d = self.calculate_centroids(MNIST_array, closest_centroid, centroids)
                SSE.append(sum(all_cluster_d))            
            all_SSE.append(min(SSE))
            all_k.append(k)
            
        #Plot the values
        plt.figure()
        plt.plot(all_SSE , all_k,'rx-')
        plt.xlabel('SSE')
        plt.ylabel('K-values')
        plt.title('The Elbow Method showing the optimal k - value')
        plt.show()        
    
    #Function for k-means clustering
    def main_kmeans(self):   
        MNIST_array, MNIST_df = self.read_data()                 
        
        #number of clusters
        k = self.k
        
        #Randomly select k number of points as centroids
        centroid_index = random.sample(range(0, len(MNIST_df)), k)    
        centroids = list()
        for i in centroid_index:
            centroids.append(MNIST_array[i])
        
        #converting list into numpy array
        centroids = np.array(centroids)    
        
        #List for sum of squared errors
        SSE = list()
        no_of_iterations = 50
        closest_centroid = list()
        for i in range(no_of_iterations):
            #closest_centroid = self.clustering_euclidean_distance(MNIST_array, centroids)
            #closest_centroid = self.clustering_cosine_similarity(MNIST_array, centroids)
            closest_centroid = self.clustering_manhattan_distance(MNIST_array, centroids)
            centroids, all_cluster_d = self.calculate_centroids(MNIST_array, closest_centroid, centroids)
            SSE.append(sum(all_cluster_d))        
        clustered_MNIST_df = pd.concat([pd.DataFrame(MNIST_array), pd.DataFrame(closest_centroid, columns=['Cluster'])], axis=1)
        clustered_MNIST_df.replace({0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10}, inplace=True)
        
        #To visualize the number iterations on kmeans and SSE
        self.visualize_sse(no_of_iterations, SSE)
                
        
        #Saving the results into the file
        clustered_MNIST_df.to_csv('MNIST_results.csv',columns=['Cluster'], index =False, header = False)
        
#CLASS END=====================================================================================================================


#MAIN START=====================================================================================================================

#Execution start time
start_time = time.time()

kmeans_obj = kmeans(k = 10)
kmeans_obj.main_kmeans()

#To visualize the different k values and SSE
#kmeans_obj.visualize_k_sse()

print("Total execution time :", time.time() - start_time, "seconds")

#MAIN END=====================================================================================================================
