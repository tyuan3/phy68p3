import numpy as np

def compute_acf(time_series, max_lag):
    """
    Computes the Autocorrelation Function (ACF) for a given 1D time series.
    
    Parameters:
        time_series (array-like): The sequential states or positions from your walk.
        max_lag (int): The maximum time displacement (lag) to calculate.
        
    Returns:
        numpy.ndarray: ACF values ranging from lag 0 to max_lag.
    """
    data = np.asarray(time_series)
    n = len(data)
    
    # 1. Center the data by subtracting the mean
    centered_data = data - np.mean(data)
    
    # 2. Calculate the variance (denominator of the ACF)
    variance = np.var(data)
    
    # Handle the edge case of a flat/unchanging line to prevent division by zero
    if variance == 0:
        return np.ones(max_lag + 1)
        
    # 3. Compute autocovariance for each lag, then normalize by variance
    acf_values = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            # Overlap the series with its lagged self
            covariance = np.mean(centered_data[:-lag] * centered_data[lag:])
            acf_values.append(covariance / variance)
            
    return np.array(acf_values)

# --- Fast Fourier Transform (FFT) Alternative for Long Time Series ---
def compute_acf_fft(time_series, max_lag):
    """
    Computes ACF using the Wiener-Khinchin theorem via FFT. 
    Highly recommended if your walk has >10,000 steps.
    """
    data = np.asarray(time_series)
    n = len(data)
    centered = data - np.mean(data)
    
    # Pad to power of 2 for FFT speed and to avoid periodic wrap-around artifacts
    next_pow2 = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_result = np.fft.fft(centered, n=next_pow2)
    
    # Power spectral density
    psd = np.abs(fft_result) ** 2
    
    # Inverse FFT to get autocovariance
    autocov = np.fft.ifft(psd).real[:n]
    
    # Normalize by the lag-0 variance
    acf = autocov / autocov[0]
    return acf[:max_lag + 1]
