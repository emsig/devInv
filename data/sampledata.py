import emg3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def create_data(name, simulation, model_init, min_offset=0):
    """Create observed and start data; store to disk."""

    # Compute observed data
    model = simulation.model.copy()
    simulation.compute(observed=True, min_offset=min_offset)
    simulation.clean('computed')

    # Replace model
    simulation.model = model_init

    # Compute start data
    simulation.compute()
    simulation.survey.data['start'] = simulation.survey.data.synthetic
    simulation.clean('computed')

    emg3d.save(f"{name}.h5", simulation=simulation, model=model)


def load_data(name):
    """Load simulation and true model."""
    data = emg3d.load(f"{name}.h5")
    return data['simulation'], data['model']


def plot_obs_initial(name):
    """Plot observed and true data."""
    sim = emg3d.load(f"{name}.h5")["simulation"]
    fig, axs = plt.subplots(
            2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)

    axs[0].plot(np.abs(sim.data.observed.squeeze()), "k", label="Observed")
    axs[0].plot(np.abs(sim.data.start.squeeze()), "C0.", label="Initial Model")

    rms = 100*np.abs((sim.data.start.squeeze() - sim.data.observed.squeeze()))
    rms /= np.abs(sim.data.observed.squeeze())
    axs[1].plot(rms, "C1", label="RMS")

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend()
    axs[1].legend()


def plot_models(sim, mstart, mtrue, zind=1):
    depth = np.round(mstart.grid.cell_centers_z[zind], 2)
    print(f"Depth slice: {depth} m")

    popts1 = {'cmap': 'Spectral_r', 'norm': LogNorm(vmin=0.1, vmax=1000)}
    # popts2 = {'edgecolors': 'grey', 'linewidth': 0.5, 'cmap': 'Spectral_r',
    #           'norm': LogNorm(vmin=0.1, vmax=1000)}
    opts = {'v_type': 'CC', 'normal': 'Y'}

    rec_coords = sim.survey.receiver_coordinates()
    src_coords = sim.survey.source_coordinates()

    fig, axs = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axs

    grid = sim.model.grid

    # True model
    out1, = grid.plot_slice(1/mtrue.property_x.ravel('F'), ax=ax1,
                            pcolor_opts=popts1, **opts)
    ax1.set_title("True Model (Ohm.m)")
    ax1.plot(rec_coords[0], rec_coords[2], 'bv')
    ax1.plot(src_coords[0], src_coords[2], 'r*')

    # Start model
    out2, = grid.plot_slice(
            1/mstart.property_x.ravel('F'), ax=ax2, pcolor_opts=popts1, **opts)
    ax2.set_title("Start Model (Ohm.m)")

    # Final inversion model
    out3, = grid.plot_slice(
            1/sim.model.property_x.ravel('F'), ax=ax3, pcolor_opts=popts1,
            **opts)
    ax3.set_title("Final Model (Ohm.m)")

    opts['normal'] = 'Z'
    opts['ind'] = zind

    # True model
    out4, = grid.plot_slice(
            1/mtrue.property_x.ravel('F'), ax=ax4, pcolor_opts=popts1, **opts)
    ax4.set_title("True Model (Ohm.m)")
    ax4.plot(rec_coords[0], rec_coords[1], 'bv')
    ax4.plot(src_coords[0], src_coords[1], 'r*')

    # Start model
    out5, = grid.plot_slice(
            1/mstart.property_x.ravel('F'), ax=ax5, pcolor_opts=popts1, **opts)
    ax5.set_title("Start Model (Ohm.m)")

    # Final inversion model
    out6, = grid.plot_slice(
            1/sim.model.property_x.ravel('F'), ax=ax6, pcolor_opts=popts1,
            **opts)
    ax6.set_title("Final Model (Ohm.m)")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('')
        ax.set_xticklabels([])

    for ax in [ax2, ax3, ax5, ax6]:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    for ax in axs.ravel():
        ax.axis('equal')

    plt.colorbar(
            out1, ax=axs, orientation='horizontal', fraction=.1, shrink=.8,
            aspect=30)


def plot_responses(sim):
    fig, axs = plt.subplots(
            2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
    real = [int(k[2:]) for k in sim.data.keys() if k.startswith('it')]
    for i in real:
        n = f"it{i}"
        axs[0].plot(np.abs(sim.data[n].squeeze()), f"C{i % 10}-", label=n)
        rms = 100*np.abs((sim.data[n].squeeze() - sim.data.observed.squeeze()))
        rms /= np.abs(sim.data.observed.squeeze())
        axs[1].plot(rms, f"C{i % 10}-")
    axs[0].plot(np.abs(sim.data.observed.squeeze()), "k.")
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend()
