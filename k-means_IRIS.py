import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances


#CLASS START=====================================================================================================================
class kmeans:
    
    def __init__(self,k):
        self.k = k
        
    #Function to read the data    
    def read_data(self):
        iris_df = pd.read_table("image_new_test_IRIS.txt", header=None, skip_blank_lines=False, delim_whitespace=True)
        iris_array = np.array(iris_df)
        return iris_array, iris_df
    
    #Function to calculate the similarity
    def clustering_cosine_similarity(self,iris_array, centroids):    
        distance_matrix = cosine_similarity(iris_array, centroids)
        closest_centroids = []
        for i in range(distance_matrix.shape[0]):
            c = np.argmax(distance_matrix[i])
            closest_centroids.append(c)
        return closest_centroids
    
    #Function to calculate euclidean distance    
    def clustering_euclidean_distance(self, iris_array, centroids):    
        distance_matrix = euclidean_distances(iris_array, centroids)
        closest_centroids = []
        for i in range(distance_matrix.shape[0]):
            c = np.argmin(distance_matrix[i])
            closest_centroids.append(c)
        return closest_centroids
    
    #Function to clculate the centroids
    def calculate_centroids(self, iris_array, nearest_centroid, centroids):
        cluster_d = list()
        all_cluster_d = [0.0,0.0,0.0]
        new_centroids = list()
        new_df = pd.concat([pd.DataFrame(iris_array), pd.DataFrame(nearest_centroid, columns=['Cluster'])], axis=1)    
        new_df_arr = np.array(new_df['Cluster'])
        for c in set(new_df_arr):        
            thiscluster = new_df[new_df['Cluster'] == c][new_df.columns[:-1]]
            temp = np.array(centroids[c])
            temp = temp.reshape(1,-1)
            cluster_d = euclidean_distances(thiscluster, temp)
            for d in cluster_d:
                all_cluster_d[c] += d*d        
            cluster_mean = thiscluster.mean(axis=0)
            new_centroids.append(cluster_mean)
        return new_centroids, all_cluster_d
    
    #Function to visualize the SSE and no.of iterations
    def visualize_sse(self, iterations, SSE):
        plt.figure()
        plt.plot(range(iterations), SSE, 'rx-')
        plt.xlabel('No.of iterations')
        plt.ylabel('SSE(Sum of squared errors)')
        plt.title('Elbow Method showing the optimal iterations')
        plt.show()
        
    
    #Function for k-means clustering
    def main_kmeans(self):   
        iris_array, iris_df = self.read_data()         
        
        #number of clusters
        k = self.k
        #Randomly select three points as centroids
        centroid_index = random.sample(range(0, len(iris_df)), k)    
        centroids = list()
        for i in centroid_index:
            centroids.append(iris_array[i])
        
        #converting list into numpy array
        centroids = np.array(centroids)    
        
        #List for sum of squared errors
        SSE = list()
        no_of_iterations = 10
        closest_centroid = list()
        for i in range(no_of_iterations):
            #closest_centroid = self.clustering_euclidean_distance(iris_array, centroids)
            closest_centroid = self.clustering_cosine_similarity(iris_array, centroids)
            centroids, all_cluster_d = self.calculate_centroids(iris_array, closest_centroid, centroids)
            SSE.append(sum(all_cluster_d))        
        clustered_iris_df = pd.concat([pd.DataFrame(iris_array), pd.DataFrame(closest_centroid, columns=['Cluster'])], axis=1)
        clustered_iris_df.replace({0:1,1:2,2:3}, inplace=True) 
        
        #To visualize the number iterations on kmeans and SSE
        self.visualize_sse(no_of_iterations, SSE)
        
        #To visualize the different values of k (clusters) and SSE
        #self.visualize_diff_kvalues(SSE)
        
        #Saving the results into the file
        clustered_iris_df.to_csv('iris_results.csv',columns=['Cluster'], index =False, header = False)
        
#CLASS END=====================================================================================================================


#MAIN START=====================================================================================================================
#Execution start time
start_time = time.time()

kmeans_obj = kmeans(k = 3)
kmeans_obj.main_kmeans()

print("Total execution time :", time.time() - start_time, "seconds")
#MAIN END=====================================================================================================================
