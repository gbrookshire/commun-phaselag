import numpy as np
import matplotlib.pyplot as plt
import simulate
import pdc

plt.ion()


def test_pdc():
    # Low-freq 'modulator' frequencies
    lf_centers = np.arange(5, 15)
    lf_bandwidth = lf_centers / 2.5  # ~4 cycles

    # High-freq 'carrier' frequencies
    hf_centers = np.arange(50, 150, 10)
    hf_bandwidth = hf_centers / 3  # ~5 cycles

    # Parameters for the simulated signals
    sim_params = dict(dur=100,
                      fs=400,
                      noise_amp=1.5)

    # Parameters for the MI phase-lag analysis
    mi_params = dict(fs=sim_params['fs'],
                     lf_centers=lf_centers,
                     lf_bandwidth=lf_bandwidth,
                     hf_centers=hf_centers,
                     hf_bandwidth=hf_bandwidth,
                     lag=6,
                     n_bins=2**3)

    # Simulate the signal
    t, s_a, s_b = simulate.sim(**sim_params)

    # Compute the phase-dependent communication
    res = pdc.pdc(s_a, s_b, **mi_params)

    # Plot it
    x = res['PD(AB)-PD(BA)']
    maxabs = np.max(np.abs(x))
    cb_ticks = [-maxabs, 0, maxabs]
    vmin = -maxabs
    cmap = plt.cm.RdBu_r
    plt.imshow(x.T,
               origin="lower",
               interpolation="none",
               aspect='auto',
               vmin=vmin, vmax=maxabs,
               cmap=cmap)
    ytick_spacing = 4
    xtick_spacing = 4
    plt.yticks(range(len(mi_params['hf_centers']))[::ytick_spacing],
               mi_params['hf_centers'][::ytick_spacing])
    plt.xticks(range(len(mi_params['lf_centers']))[::xtick_spacing],
               mi_params['lf_centers'][::xtick_spacing])
    plt.xlabel('LF freq (Hz)')
    plt.ylabel('HF freq (Hz)')
    cb = plt.colorbar(format='%.1e')
    cb.set_ticks(cb_ticks)
    cb.ax.set_ylabel('bits $^2$ / Hz')
    plt.ylabel('HF freq (Hz)')
    plt.xlabel('Phase freq (Hz)')


if __name__ == '__main__':
    test_pdc()
