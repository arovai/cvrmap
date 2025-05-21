import numpy as np
from formulaic.transforms.cubic_spline import cubic_spline
from nilearn import masking
import peakutils
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline

from cvrmap.utils.loaders import (load_roi_mask,
                                  load_bold_img,
                                  load_co2_data)
from cvrmap.utils.writers import write_timeseries_data_and_figure


def compute_roi_signal(layout, sub, mask, roi_name, config):
    bold, config = load_bold_img(layout, sub, config)
    roi_data = masking.apply_mask(bold, mask)
    roi_data = np.nanmean(roi_data, axis=(0,1,2))

    return write_timeseries_data_and_figure(layout,
                                            sub,
                                            roi_name,
                                            f"Mean BOLD signal in {roi_name}",
                                            roi_data,
                                            config)


def compute_upper_envelope(data, metadata, config):
    # Apply gaussian filtering to stabilize peak localisation
    sigma_for_peak_localisation = config["sigma_for_peak_localisation"]*metadata["SamplingFrequency"]
    data_for_peak_localisation = gaussian_filter(data, sigma_for_peak_localisation)

    # Extract peaks indexes
    peaks_idx = peakutils.indexes(data_for_peak_localisation)

    # Create interpolator class based on original data
    interpolator = CubicSpline(peaks_idx, data)

    # Restrict time span corresponding to first to last peak
    first_peak_idx = peaks_idx[0]
    last_peak_idx = peaks_idx[1]
    span = np.arange(0, data.shape[0])
    restricted_span = span[first_peak_idx:last_peak_idx]

    # Find interpolated data on relevant time span
    upper_envelope_data = interpolator(restricted_span)

    return upper_envelope_data, metadata


def compute_baseline(data):
    # np.mean(baseline_data)*np.ones(len(baseline_data))
    return np.mean(peakutils.baseline(data))


def smooth_timeseries(data, sampling_frequency, config):
    sigma_seconds = config["physio_smoothing_sigma"]
    sigma = sigma_seconds*sampling_frequency
    print(f"Applying gaussian smoothing with sigma of"
          f"of {sigma_seconds} seconds.")
    return gaussian_filter(data, sigma)


def compute_etco2(layout, sub, config):
    co2_data, co2_metadata = load_co2_data(layout, sub, config)
    sampling_frequency = co2_metadata["SamplingFrequency"]
    write_timeseries_data_and_figure(layout,
                                     sub,
                                     name="co2",
                                     description="Raw CO2 recording",
                                     data=co2_data,
                                     sampling_frequency=sampling_frequency)

    smooth_co2_data = smooth_timeseries(co2_data, sampling_frequency, config)

    etco2_data, etco2_metadata = compute_upper_envelope(smooth_co2_data, co2_metadata, config)

    write_timeseries_data_and_figure(layout,
                                     sub,
                                     name="etco2",
                                     description="Extracted end-tidal CO2",
                                     data=etco2_data,
                                     sampling_frequency=sampling_frequency)
    return etco2_data, etco2_metadata


def compute_probe_regressor(layout, sub, config):
    if config["roi"]:
        roi_name, mask = load_roi_mask(layout, sub, config)
        return compute_roi_signal(layout, sub, mask, roi_name, config)
    else:
        return compute_etco2(layout, sub, config)
