import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Directory where heatmaps will be saved
    output_dir = 'local_density_of_states_heatmap'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop over file indices from 0 to 10
    for i in range(11):
        # Construct the filename
        filename = f'local_density_of_states_for_level_{i}.txt'
        
        # Check if the file exists
        if not os.path.isfile(filename):
            print(f'Warning: {filename} not found. Skipping.')
            continue
        
        # Load the data as a 2D array
        data = np.genfromtxt(filename, delimiter=',', comments=None, 
                     filling_values=np.nan, invalid_raise=False)
        
        # Create a figure for the heatmap
        plt.figure()
        
        # Plot the heatmap
        # - origin='lower' places the first row of data at the bottom of the y-axis
        # - cmap can be changed to any preferred colormap (e.g., 'plasma', 'inferno')
        plt.imshow(data, cmap='viridis', origin='lower')
        
        # Add a color bar with label
        cbar = plt.colorbar()
        cbar.set_label('Local Electron Density', rotation=90)
        
        # Label the plot
        plt.title(f'LDOS Heatmap for Level {i}')
        plt.xlabel('X (grid index)')
        plt.ylabel('Y (grid index)')
        
        # Save the figure with a descriptive filename
        out_fig_name = os.path.join(output_dir, f'ldos_heatmap_level_{i}.png')
        plt.savefig(out_fig_name, dpi=300, bbox_inches='tight')
        
        # Close the figure to free up memory
        plt.close()

if __name__ == '__main__':
    main()

