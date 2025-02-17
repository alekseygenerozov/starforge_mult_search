from collections import defaultdict
import numpy as np

def npz_stack(npz_list):
    # Dictionary to hold lists of arrays for each key
    data_dict = defaultdict(list)

    # Loop over .npz files in the directory
    for filename in npz_list:
        # Load the .npz file
        data = np.load(filename, allow_pickle=True)

        # Iterate over keys in the .npz file
        for key in data.keys():
            data_dict[key].append(data[key])  # Append data for this key
    concatenated_data_dict = {key: np.concatenate(arrays) for key, arrays in data_dict.items()}

    return concatenated_data_dict