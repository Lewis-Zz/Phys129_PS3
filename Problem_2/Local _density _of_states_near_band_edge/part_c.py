import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Sub-region definitions (example)
    row_start, row_end = 10, 30  # will slice rows 10 through 29
    col_start, col_end = 20, 40  # will slice columns 20 through 39

    # Store average LDOS for each file in a list
    avg_ldos_values = []

    # We assume files from index 0 to 10
    file_indices = range(11)

    for i in file_indices:
        filename = f'local_density_of_states_for_level_{i}.txt'
        
        if not os.path.isfile(filename):
            print(f"Warning: {filename} not found. Skipping.")
            avg_ldos_values.append(np.nan)
            continue
        
        # Load data (adjust delimiter if needed, e.g., delimiter=',' for CSV)
        data = np.genfromtxt(filename, delimiter=',', comments=None, 
                     filling_values=np.nan, invalid_raise=False) 
        # Extract the sub-region:
        subregion = data[row_start:row_end, col_start:col_end]
        
        # Compute the average LDOS in this sub-region
        mean_ldos = np.mean(subregion)
        
        avg_ldos_values.append(mean_ldos)

    # Plot the average LDOS vs. file index
    plt.figure()
    plt.plot(file_indices, avg_ldos_values, marker='o', linestyle='-', color='b')
    plt.title('Average LDOS in Sub-Region vs. Level Index')
    plt.xlabel('File/Level Index')
    plt.ylabel('Average LDOS (Sub-Region)')
    plt.grid(True)

    # Save or show the plot
    plt.savefig('average_ldos_subregion.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

