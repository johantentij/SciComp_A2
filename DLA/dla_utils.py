from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_gif(frames_field, frames_cluster):
    """
        Generates and saves a GIF animation of the DLA growth process.

        This function overlays the growing cluster on top of the evolving
        Laplacian potential field using two distinct layers for clarity.

        Parameters:
        -----------
        frames_field : list of numpy.ndarray
            History of the potential field arrays.
        frames_cluster : list of numpy.ndarray
            History of the boolean cluster masks.
    """

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95)

    # Layer 1: The Potential Heatmap
    im_field = ax.imshow(frames_field[0].T, origin='lower', cmap='cividis', vmin=0, vmax=1)

    # Layer 2: The Visible Seed (White)
    # We use a custom cmap where 0 is transparent and 1 is opaque white

    cmap_tree = ListedColormap([[0, 0, 0, 0], [1, 1, 1, 1]])
    im_cluster = ax.imshow(frames_cluster[0].T, origin='lower', cmap=cmap_tree, interpolation='nearest')

    # ax.axis('off')
    title = ax.set_title("DLA: Optimized Growth (Boundary Search + Warm Start)", color='white', fontsize=14)

    def update(i):
        im_field.set_array(frames_field[i].T)
        im_cluster.set_array(frames_cluster[i].T)
        return [im_field, im_cluster]

    ani = animation.FuncAnimation(fig, update, frames=len(frames_field), interval=30, blit=True)

    filename = "dla_fully_optimized.gif"
    ani.save(filename, writer="pillow", fps=30)
    plt.close(fig)
    print(f"Saved: {filename}")

def plot_last_state(frames_field, frames_cluster):
    """
        Renders a high-quality static plot of the final simulation state.

        Parameters:
        -----------
        frames_field : list of numpy.ndarray
            Snapshot history of the field (only the last index is used).
        frames_cluster : list of numpy.ndarray
            Snapshot history of the cluster (only the last index is used).

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object for the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the rendered field and cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')

    ax.axis('off')
    ax.grid(False)

    ax.imshow(frames_field.T, origin='lower', cmap='cividis', vmin=0, vmax=1)

    # Layer 2: The Visible Seed (White)
    # We use a custom cmap where 0 is transparent and 1 is opaque white

    cmap_tree = ListedColormap([[0, 0, 0, 0], [1, 1, 1, 1]])
    ax.imshow(frames_cluster.T, origin='lower', cmap=cmap_tree, interpolation='nearest')

    ax.set_title("DLA: Optimized Growth (Boundary Search + Warm Start)", color='white', fontsize=14)
    return fig, ax