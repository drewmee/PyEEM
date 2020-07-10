from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


def abs_pick_peaks():
    return


def spectral_slopes():
    return


def spectral_slope_ratio():
    return


# Fluorescence


def fluorescence_index():
    return


def humification_index():
    return


def freshness_index():
    return


def relative_fl_index():
    return


def biological_index():
    return


def fl_pick_peaks():
    return


def fl_peak_ratio():
    return


def find_peaks(eem):
    """
    Takes an image and detect the peaks using
    the local maximum filter. Returns a boolean mask 
    of the peaks (i.e. 1 when the pixel's value is 
    the neighborhood maximum, 0 otherwise)
    """

    # https://stackoverflow.com/questions/53236058/find-2d-peak-prominence-using-python
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 1)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(eem, 5) == eem
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = eem == 0

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(eem)
    plt.imshow(detected_peaks)
    plt.show()
    return detected_peaks
