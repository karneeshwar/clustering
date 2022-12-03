# To compute the quantization error in clustering
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


# Function: main
#   checks for arguments, imports data and computes quantization error
if __name__ == '__main__':
    # If the number of arguments are incorrect prompt user to rerun program and exit
    if len(system.argv) != 5:
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

    # Declare and initialize required variables for clustering
    datas, features = input_data.shape
    k_clusters = int(system.argv[2])
    r_iterations = int(system.argv[3])
    initial_labels = numpython.zeros(datas, dtype='int')
    quant_error = float('inf')

    # Running the algorithm for r_iterations based on the input arguments
    while r_iterations > 0:
        # Decrement the number of r_iterations each time
        r_iterations -= 1

        # Randomly select a row and apply kmeans++ initialization algorithm to determine the rest of the cluster means
        initializer = 1
        cluster_mean = {initializer: input_data[numpython.random.choice(range(0, datas))]}
        # Run until we have k means for clusters k clusters
        while k_clusters > initializer:
            initializer += 1
            dist_dr = 0
            weights = [0]*datas
            # Compute distance of each data point from the cluster means
            for data in range(datas):
                dist_nr = float('inf')
                # Compute the minimum distance from all the cluster means
                for cluster in cluster_mean.keys():
                    dist_nr = min(numpython.sum(numpython.square(input_data[data] - cluster_mean[cluster])), dist_nr)
                # The weights are proportional to the distance square of each data point from the cluster means
                dist_dr += dist_nr ** 2
                weights[data] = dist_nr ** 2
            weights /= dist_dr
            # Randomly choose the next cluster mean using the calculated weights
            cluster_mean[initializer] = input_data[numpython.random.choice(range(0, datas), p=weights)]

        # Run the clustering algorithm until convergence
        while 1:
            # Compute the minimum distance of each point from each cluster and assign the cluster that it is closest to
            current_labels = numpython.zeros(datas, dtype='int')
            for data in range(datas):
                dist_min = float('inf')
                for cluster in cluster_mean.keys():
                    current_dist = numpython.sum(numpython.square(input_data[data] - cluster_mean[cluster]))
                    if dist_min >= current_dist:
                        current_labels[data], dist_min = cluster, current_dist

            # If the currently assigned clusters are not same as previous one then we have not reached convergence
            # Compute the new cluster means and continue with the loop
            if not numpython.array_equal(initial_labels, current_labels):
                cluster_mean = compute_means(input_data, current_labels)
                initial_labels = current_labels
            # If the currently assigned clusters are same as previous one then we reached convergence
            # Compute the quantization error and break the loop
            else:
                current_qe = compute_qe(input_data, current_labels, cluster_mean)
                break

        # Check if the previous quantization error is greater than the currently computed one
        # Then the computed qe is the new minimum and the corresponding labels are stored
        # Continue with r_iterations, at the end of it the final clusters with the least error is obtained
        if quant_error > current_qe:
            output_labels, quant_error = initial_labels, current_qe

    # Exception handling for output label file
    while 1:
        try:
            numpython.savetxt(system.argv[4], output_labels, fmt='%d')
            break
        except IOError:
            print('File not written')

    # Print the quantization error in console
    print(quant_error)
