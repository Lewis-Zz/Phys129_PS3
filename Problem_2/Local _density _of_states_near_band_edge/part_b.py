import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def main():
    # Name of the subfolder where we'll save the height profile plots
    output_dir = 'local_density_of_states_height'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of files, assuming they are from level 0 to level 10
    for i in range(11):
        # Construct filename
        filename = f'local_density_of_states_for_level_{i}.txt'
        
        # Check if file exists
        if not os.path.isfile(filename):
            print(f'Warning: {filename} not found. Skipping.')
            continue
        
        # Load data (modify 'delimiter' if necessary, e.g. delimiter=',' for CSV)
        data = np.genfromtxt(filename, delimiter=',', comments=None, 
                     filling_values=np.nan, invalid_raise=False) 
        # Create X, Y grids for surface plotting
        # - data.shape gives (num_rows, num_cols)
        num_rows, num_cols = data.shape
        X = np.arange(num_cols)
        Y = np.arange(num_rows)
        X, Y = np.meshgrid(X, Y)
        Z = data  # Z is the LDOS values
        
        # Create the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        # Optional: add a colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, label='Local Density of States')
        
        # Label the axes
        ax.set_title(f'LDOS 3D Surface - Level {i}')
        ax.set_xlabel('X (grid index)')
        ax.set_ylabel('Y (grid index)')
        ax.set_zlabel('LDOS (height)')
        
        # Save figure to the output directory
        out_fig = os.path.join(output_dir, f'ldos_surface_level_{i}.png')
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close(fig)

if __name__ == '__main__':
    main()

