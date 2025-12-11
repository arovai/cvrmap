"""
Cross-correlation utilities for CVR analysis.

This module provides functions for cross-correlating physiological probe signals
with BOLD timecourses and other signals for CVR analysis.
"""
import numpy as np  # Keep this here to improve performance

def cross_correlate(reference_container, shifted_probes_data, logger=None, config=None):
    """
    Cross-correlate a reference signal with multiple shifted probe signals.
    
    Parameters:
    -----------
    reference_container : DataContainer
        Container with the reference signal (e.g., normalized IC timecourse)
    shifted_probes_data : tuple
        Tuple containing:
        - shifted_signals: 2D numpy array (n_delays, n_timepoints)
        - time_delays_seconds: 1D numpy array of time delays in seconds
    logger : Logger, optional
        Logger instance for debugging
    config : dict, optional
        Configuration dictionary (not used in current implementation)
    
    Returns:
    --------
    tuple
        - best_correlation: float, highest correlation found
        - best_delay_seconds: float, delay in seconds corresponding to best correlation
    """
    
    best_correlation = 0.0
    best_delay_seconds = None
    
    reference_signal = reference_container.data

    # Unpack the shifted probes data
    shifted_signals, time_delays_seconds = shifted_probes_data

    if shifted_signals is None or time_delays_seconds is None:
        return 0.0, 0.0

    n_delays = shifted_signals.shape[0]

    # DEBUG: Show reference signal info
    if logger:
        logger.debug(f"DEBUG XCORR: Reference signal - shape: {reference_signal.shape}, mean: {np.mean(reference_signal):.6f}, std: {np.std(reference_signal):.6f}")
        logger.debug(f"DEBUG XCORR: Reference signal - first 10 points: {reference_signal[:10]}")
        logger.debug(f"DEBUG XCORR: Shifted signals - shape: {shifted_signals.shape}, n_delays: {n_delays}")

        # Find zero-delay index for comparison
        zero_delay_idx = np.argmin(np.abs(time_delays_seconds))
        logger.debug(f"DEBUG XCORR: Zero-delay index: {zero_delay_idx}, delay: {time_delays_seconds[zero_delay_idx]:.1f}s")
        zero_delay_signal = shifted_signals[zero_delay_idx, :]
        logger.debug(f"DEBUG XCORR: Zero-delay probe signal - shape: {zero_delay_signal.shape}, mean: {np.mean(zero_delay_signal):.6f}, std: {np.std(zero_delay_signal):.6f}")
        logger.debug(f"DEBUG XCORR: Zero-delay probe signal - first 10 points: {zero_delay_signal[:10]}")

        # Check if signals are identical at zero delay
        if reference_signal.shape == zero_delay_signal.shape:
            signal_diff = np.abs(reference_signal - zero_delay_signal)
            logger.debug(f"DEBUG XCORR: Difference at zero delay - max: {np.max(signal_diff):.10f}, mean: {np.mean(signal_diff):.10f}")

    for i in range(n_delays):
        # Both signals are normalized and of same length, so correlation is simply dot product / n_samples
        probe_signal = shifted_signals[i, :]

        # Calculate correlation for normalized signals: dot product / n_samples
        # Use float64 (double precision) to avoid rounding errors
        correlation = np.float64(np.dot(reference_signal, probe_signal)) / np.float64(len(reference_signal))

        # Debug: log high correlations with high precision
        if logger and correlation > 0.5:
            logger.debug(f"High correlation found: i={i}, delay={time_delays_seconds[i]:.1f}s, corr={correlation:.15f}")

        if correlation > best_correlation:
            best_correlation = correlation
            best_delay_seconds = time_delays_seconds[i]
    
    if best_delay_seconds is None:
        best_delay_seconds = 0.0
    
    return best_correlation, best_delay_seconds