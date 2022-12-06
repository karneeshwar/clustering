# To perform Fuzzy c-means clustering with K-means++ initialization
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython
numpython.random.seed(12)


# Function: compute_qe
# Parameters: matrix = input data, groups = input label, data_mean = mean respective to each cluster
#   To compute the quantization error in clustering
def compute_qe(matrix, groups, data_mean):
    # Compute the quantization error
    rows, columns = matrix.shape
    q_error = 0
    for row in range(rows):
        current_label = groups[row]
        current_mean = data_mean[current_label]
        q_error += numpython.sum(numpython.square(matrix[row, :] - current_mean))

    # Return the quantization error
    return q_error


# Function: compute_means
# Parameters: matrix = input data, groups = input label
#   To compute the mean point of each cluster
def compute_means(matrix, groups):
    # Compute unique labels, count and sum of data corresponding to each label
    rows, columns = matrix.shape
    unique_groups = numpython.unique(groups)
    group_count = unique_groups.shape[0]
    group_sum = numpython.zeros([group_count, columns])
    group_size = numpython.zeros([1, group_count])
    group_index = {}
    group_counter = 0

    for each_label in unique_groups:
        group_index[each_label] = group_counter
        group_counter += 1
        for row in range(rows):
            if groups[row] == each_label:
                for col in range(columns):
                    group_sum[group_index[each_label], col] += matrix[row, col]
                group_size[0, group_index[each_label]] += 1

    # Compute the mean of data in each label and store it in a dictionary
    data_mean = {}
    for row in range(group_sum.shape[0]):
        data_mean[unique_groups[row]] = group_sum[row, :] / group_size[0, row]

    # Return the new cluster mean
    return data_mean


# Function: compute_fuzzy_qe
# Parameters: groups = input label, p = input value of p, d_matrix = d(i, j)
#   To compute the quantization error in clustering
def compute_fuzzy_qe(groups, p, d_matrix):
    # Compute the quantization error
    rows, columns = groups.shape
    fuzzy_q_error = 0
    for row in range(rows):
        for column in range(columns):
            fuzzy_q_error += (groups[row][column]**p) * dist[row][column]

    # Return the quantization error
    return fuzzy_q_error


# Function: compute_fuzzy_means
# Parameters: matrix = input data, groups = input label, k_labels = num of clusters, p = input value of p
#   To compute the mean point of each cluster
def compute_fuzzy_means(matrix, groups, k_labels, p):
    # Compute the new means for each cluster based on c(i, j) = groups, xi = input data and p
    rows, columns = matrix.shape
    data_mean = {}
    for each_label in range(k_labels):
        mean_nr = 0
        mean_dr = 0
        for row in range(rows):
            mean_nr += (groups[row][each_label]**p) * matrix[row]
            mean_dr += groups[row][each_label]**p
        data_mean[each_label+1] = mean_nr/mean_dr

    # Return the new cluster mean
    return data_mean


# Function: hard_clustering
# Parameters: groups = input label
#   To convert the soft clustering labels to hard clustering labels
def hard_clustering(groups):
    # Assign each data point to an individual cluster based on it's value in the soft clustering groups
    # Higher the value in soft clustering, then that data point is completely assigned to that cluster in hard clustering
    rows, columns = groups.shape
    hard_groups = numpython.zeros(rows, dtype='int')
    for row in range(rows):
        max_value = float('-inf')
        for column in range(columns):
            if groups[row][column] > max_value:
                hard_groups[row] = column+1
                max_value = groups[row][column]

    # Return the hard clustering groups
    return hard_groups


# Function: main
#   checks for arguments, imports data and computes quantization error
if __name__ == '__main__':
    # If the number of arguments are incorrect prompt user to rerun program and exit
    if len(system.argv) != 6:
        print('Incorrect arguments, rerun the program with correct number of arguments!')
        system.exit()

    # Exception handling for input files
    while 1:
        try:
            input_data = numpython.genfromtxt(system.argv[1], delimiter=',', autostrip=True)
            break
        except IOError:
            print('Input data file not found')
            system.exit()

    # # Homework Testing
    # input_data = numpython.array([[0, 0], [4, 0], [5, 1], [6, 0]])

    # Variable for determining the convergence of soft clustering
    # It is a percent value of how much deviation it can have from previous labels => + or - c_value%
    # The default value assigned below is 0.01% = 0.0001
    # This value can be changed which will reflect in line 195
    c_value = 0.0001

    # Declare and initialize required variables for clustering
    datas, features = input_data.shape
    k_clusters = int(system.argv[2])
    r_iterations = int(system.argv[3])
    p_value = int(system.argv[4])
    if p_value < 0 or p_value == 1:
        print("p is any positive integer but 1, rerun with correct value for p")
        exit(0)
    initial_labels = numpython.zeros((datas, k_clusters))
    fuzzy_quant_error = float('inf')

    # Running the algorithm for r_iterations based on the input arguments
    while r_iterations > 0:
        # Decrement the number of r_iterations each time
        r_iterations -= 1

        # Randomly select a row and apply kmeans++ initialization algorithm to determine the rest of the cluster means
        initializer = 1
        cluster_mean = {initializer: input_data[numpython.random.choice(range(0, datas), replace=False)]}
        # Run until we have k means for k clusters
        while k_clusters > initializer:
            initializer += 1
            dist_dr = 0
            weights = [0] * datas
            # Compute distance of each data point from the cluster means
            for data in range(datas):
                dist_nr = float('inf')
                # Compute the minimum distance from all the cluster means
                for cluster in cluster_mean.keys():
                    dist_nr = min(numpython.sum(numpython.square(input_data[data] - cluster_mean[cluster])) ** 0.5, dist_nr)
                # The weights are proportional to the distance square of each data point from the cluster means
                dist_dr += dist_nr ** 2
                weights[data] = dist_nr ** 2
            weights /= dist_dr
            # Randomly choose the next cluster mean using the calculated weights
            cluster_mean[initializer] = input_data[numpython.random.choice(range(0, datas), p=weights)]

        # # Homework Testing
        # cluster_mean[1] = input_data[2]
        # cluster_mean[2] = input_data[3]

        # Run the clustering algorithm until convergence
        while 1:
            # Compute the distance of each point from each cluster mean and calculate the soft clustering labels
            current_labels = numpython.zeros((datas, k_clusters))
            dist = numpython.zeros((datas, k_clusters))
            inverted_dist = numpython.zeros((datas, k_clusters))
            for data in range(datas):
                for cluster in cluster_mean.keys():
                    dist[data][cluster - 1] = numpython.sum(numpython.square(input_data[data] - cluster_mean[cluster]))
                    if not numpython.array_equal(input_data[data], cluster_mean[cluster]):
                        inverted_dist[data][cluster - 1] = (1 / dist[data][cluster - 1]) ** (1/(p_value-1))

                inverted_dist_sum = numpython.sum(inverted_dist, axis=1)

                for cluster in cluster_mean.keys():
                    if numpython.array_equal(input_data[data], cluster_mean[cluster]):
                        for previous in range(cluster):
                            current_labels[data][previous] = 0
                        current_labels[data][cluster - 1] = 1
                        break
                    else:
                        current_labels[data][cluster - 1] = inverted_dist[data][cluster - 1] / inverted_dist_sum[data]

            # Compute the mean for each cluster labels, mean of c(i, j) column wise
            # If the change in mean of each cluster labels is just 0.01% then we reached convergence
            initial_cluster_mean = numpython.mean(initial_labels, axis=0)
            current_cluster_mean = numpython.mean(current_labels, axis=0)
            convergence = 0
            for each_cluster in range(initial_cluster_mean.shape[0]):
                if (1-c_value)*initial_cluster_mean[each_cluster] <= current_cluster_mean[each_cluster] <= (1+c_value)*initial_cluster_mean[each_cluster]:
                    convergence = 1
                else:
                    convergence = 0

            # If the currently assigned clusters are not relatively (0.01%) same as previous one then we have not reached convergence
            # Compute the new cluster means and continue with the loop
            if not convergence:
                cluster_mean = compute_fuzzy_means(input_data, current_labels, k_clusters, p_value)
                initial_labels = current_labels
            # If we reached convergence
            # Compute the quantization error and break the loop
            else:
                current_qe = compute_fuzzy_qe(current_labels, p_value, dist)
                break

        # Check if the previous quantization error is greater than the currently computed one
        # Then the computed qe is the new minimum and the corresponding labels are stored
        # Continue with r_iterations, at the end of it the final clusters with the least error is obtained
        if fuzzy_quant_error > current_qe:
            soft_labels, fuzzy_quant_error = initial_labels, current_qe

    # Convert the soft labels to hard labels
    output_labels = hard_clustering(soft_labels)

    # Call the function to compute the cluster means of the hard labels
    cluster_means = compute_means(input_data, output_labels)

    # Call the function to compute the quantization error from the hard labels
    hard_quant_error = compute_qe(input_data, output_labels, cluster_means)

    # Exception handling for output label file
    while 1:
        try:
            numpython.savetxt(system.argv[5], output_labels, fmt='%d')
            break
        except IOError:
            print('File not written')

    # Print the quantization error in console
    print('Soft clustering quantization error =', fuzzy_quant_error)
    print('Hard clustering quantization error =', hard_quant_error)
