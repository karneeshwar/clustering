# To compute the quantization error in clustering
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


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
    if len(system.argv) != 3:
        print('Incorrect arguments, rerun the program with correct number of arguments!')
        system.exit()

    # Exception handling for input files
    while 1:
        try:
            input_data = numpython.asmatrix(numpython.genfromtxt(system.argv[1], delimiter=',', autostrip=True))
            break
        except IOError:
            print('Input data file not found')
            system.exit()
    while 1:
        try:
            labels = numpython.genfromtxt(system.argv[2], delimiter=',', autostrip=True)
            break
        except IOError:
            print('Input labels file not found')
            system.exit()

    # Call the function to compute the cluster means
    cluster_means = compute_means(input_data, labels)

    # Call the function to compute the quantization error
    quant_error = compute_qe(input_data, labels, cluster_means)

    # Print the quantization error
    print(quant_error)
# 78.94506582597724