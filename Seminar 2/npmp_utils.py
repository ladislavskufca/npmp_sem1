import math
from scipy.signal import find_peaks


# INPUTS:
# sig ... 1D signal
# T ... vector of time steps
# threshold ... minimal oscillation amplitudes to threat the behaviour as oscillatory
# plot_on ... 0: no plotting, 1: plotting
#
# OUTPUTS:
# oscillatory ... 0: NO, 1: YES
# frequency
# amplitude
# spikiness
def measure_osc(sig, T, threshold):
    damped = 0

    # params
    # minDist = 1/dt;

    # [peaks,locs,~,p] = findpeaks(sig,'MinPeakProminence',threshold);
    peaks, _ = find_peaks(sig, prominence=threshold)
    # plt.plot(sig)
    # plt.plot(peaks, sig[peaks], "x")
    # plt.show()
    if peaks.any():
        if (len(peaks) >= 2):
            threshold2 = 0.1 * sig[int(math.ceil(len(peaks) / 2))]
            peaks, _ = find_peaks(sig, prominence=threshold2)
            if (peaks.any() or len(peaks) < 2):
                damped = 1

    amplitude = 0
    period = 0
    oscillatory = 0
    frequency = 0

    if peaks.any():
        if len(peaks) >= 2:
            amplitude = sig[peaks[len(peaks) - 2]] - min(sig[peaks[len(peaks) - 2]:peaks[len(peaks) - 1]])
            period = T[peaks[len(peaks) - 1]] - T[peaks[len(peaks) - 2]]
            """
            if oscillations are not damped the last peak should lie in the interval
            t_end - period (1.5*period - last peak can be misdetected)
            """
            if T[peaks[len(peaks) - 1]] < T[len(T) - 1] - 1.5 * period:
                amplitude = 0
                period = 0
                damped = 1
            else:
                frequency = 1 / period
                oscillatory = 1

    # print([oscillatory, frequency, period, amplitude, damped])
    return [oscillatory, frequency, period, amplitude, damped]


if __name__ == "__main__":
    print("npmp_utils")
