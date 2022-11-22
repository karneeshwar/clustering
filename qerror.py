# To compute the quantization error in clustering
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


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
            data = numpython.asmatrix(numpython.genfromtxt(system.argv[1], delimiter=',', autostrip=True))
            break
        except IOError:
            print('Input data file not found')
            system.exit()
    while 1:
        try:
            label = numpython.genfromtxt(system.argv[2], delimiter=',', autostrip=True, dtype='int')
            break
        except IOError:
            print('Input labels file not found')
            system.exit()

    # Compute unique labels, count and sum of data corresponding to each label
    unique_labels = numpython.unique(label)
    label_count = unique_labels.shape[0]
    rows, columns = data.shape
    label_sum = numpython.zeros([label_count, columns])
    label_size = numpython.zeros([1, label_count], dtype='int')
    label_index = {}
    label_counter = 0

    for each_label in unique_labels:
        label_index[each_label] = label_counter
        label_counter += 1
        for row in range(rows):
            if label[row] == each_label:
                for col in range(columns):
                    label_sum[label_index[each_label], col] += data[row, col]
                label_size[0, label_index[each_label]] += 1

    # Compute the mean of data in each label and store it in a dictionary
    data_mean = {}
    for row in range(label_sum.shape[0]):
        data_mean[unique_labels[row]] = label_sum[row, :]/label_size[0, row]

    # Compute the quantization error
    q_error = 0
    for row in range(rows):
        current_label = label[row]
        current_mean = data_mean[current_label]
        q_error += numpython.sum(numpython.square(data[row, :] - current_mean))

    # Print the quantization error in console
    print(q_error)
