import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def mask_4D_data(data_4D, mask_3D, nan=True):
    """apply 3D mask to 4D data"""
    
    # Ensure the data and mask have the same shape in the first three dimensions
    assert data_4D.shape[:-1] == mask_3D.shape, "data and mask shapes do not match in first three dimensions."

    # Set 3D mask to NaN wherever it is 0
    if nan == True:
        mask_3D[mask_3D == 0] = np.nan

    # Use NumPy broadcasting to apply the mask to the last dimension of the data
    data_4D_masked = data_4D * mask_3D[..., np.newaxis]

    return data_4D_masked


def binarize(mask, threshold):
    """binarize the input array based on the given threshold"""
    return np.where(mask > threshold, 1., 0.)


def check_binary(mask):
    """check if the give input is binary"""
    return np.all(np.logical_or(mask == 0, mask == 1))


def load_data(image_path, mask_path, ETL):
    """
    load nifti dataset and brain mask, return masked image as numpy data
    """
    
    image =  nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    # uncomment the next line if mask erosion is needed.
    # mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=3).astype(mask.dtype)
    mask_4d = np.repeat(mask[:, :, :, np.newaxis], ETL, axis=3)
    mask_4d[mask_4d==0] = np.nan
    masked_image = image*mask_4d
    
    return image, mask, masked_image


def flatten_filter_normalize(data_4D_masked):
    """
    flatten 4D to 2D, filter out nan rows, normalize to the first echo intensity,
    return non-normalized and normalized signals
    """
    
    # Reshape the masked array to 2D, with the last dimension flattened
    data_flat = data_4D_masked.reshape(-1, data_4D_masked.shape[-1])

    # Remove rows containing NaN values
    data_flat = data_flat[~np.isnan(data_flat).any(axis=1)]
    
    # Remove rows that the first element is zero
    data_flat = data_flat[data_flat[:, 0] != 0]
    
    # Normalize the array by the first element of the last dimension
    first_elem = data_flat[:, 0]
    data_flat_norm = data_flat / first_elem[:, np.newaxis]

    return data_flat, data_flat_norm


def imshow_along_axis(*arrays, axis=0, slice_num=0):
    """
    Plots the image along a given axis at a particular slice.

    Parameters:
    *arrays: arbitrary number of n-dimensional arrays
    axis: int, the axis to plot the image along (default=0)
    slice_num: int, the slice number to plot (default=0)

    Returns:
    None
    """

    # Check that all input arrays have the same shape (use `set` to find the number of unique shapes)
    shapes = [array.shape for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError("All input arrays must have the same shape")

    # Select the slice from each array along the given axis
    slices = [np.take(array, slice_num, axis=axis) for array in arrays]

    # Create a figure with a separate axis for each slice
    num_slices = len(slices)
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
    if num_slices == 1: # handle a single plot
        axes = [axes]
    
    # Plot each slice in a separate axis
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice)
        axes[i].grid(False)
        # axes[i].set_xlabel(f"Axis {axis}")
        # axes[i].set_ylabel("Array")

    # Show the plot
    plt.show()


def plot_spectra(*spectra, basis, labels, nrow=2, ncol=4):
    """plot 2D spectral arrays (randomly pick nrow*ncol  spectra to plot)"""
    
    # Check that all input arrays have the same shape (use `set` to find the number of unique shapes)
    shapes = [spectrum.shape for spectrum in spectra]
    if len(set(shapes)) != 1:
        raise ValueError("All input arrays must have the same shape")

    ### randomly pick a few test examples and plot them
    plt.figure(figsize=(6*ncol,5*nrow))
    plt.style.use('ggplot')
    for i in range(nrow*ncol):
        plt.subplot(nrow, ncol, i+1)
        random_pick = np.random.randint(0, shapes[0][0])
        for spectrum, label in zip(spectra, labels):    
            plt.plot(basis, spectrum[random_pick,:], label=label)
            plt.legend(fontsize=12)
            plt.xscale('log')
            #plt.title('Similarity = {}'.format(similarity_score[random_pick]), fontsize=15)
    plt.show()


def plot_decay_curve(te, signals, lines):
    """plot 2D signals array (randomly pick # of lines to plot)"""
    
    # Select random indices
    indices = np.random.choice(signals.shape[0], size=lines, replace=False)

    # Get the corresponding rows of siganls
    signals_subset = signals[indices, :]

    # Plot the selected lines on a single figure
    fig, ax = plt.subplots()
    for i in range(lines):
        ax.plot(te, signals_subset[i, :])

    # Add axis labels and a title
    ax.set_xlabel('TE (ms)')
    ax.set_ylabel('Signals')
    ax.set_title('multi-echo decay curve')

    # Show the plot
    plt.show()


def amps_sum2one(array):
    """Normalize the amps array to have a sum of one in the last dimension"""
    array = np.array(array)
    total = np.sum(array, axis=-1, keepdims=True)

    # Replace zero sum with NaN
    total = np.where(total != 0, total, np.nan)

    return array / total


def mwf_production(t2s, amps, cutoff):
    """produce mwf values based on basis t2s, amps and a given cutoff value"""
    
    # cast into numpy arrays
    t2s = np.array(t2s)
    amps = np.array(amps)
    
    # normalize amps along the last dimension
    amps = amps_sum2one(amps)

    # Create a boolean mask indicating t2s less than the cutoff
    mask = t2s <= cutoff
    
    # Apply the mask to amps
    masked_amps = np.where(mask, amps, 0)
    
    # Calculate the amplitude sum along the last dimension
    mwf = np.sum(masked_amps, axis=-1)
    
    return mwf
