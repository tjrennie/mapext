import numpy as np
from astropy.wcs import WCS

import mapext.simulation as m_sims
from mapext.emission import continuum


def test1():
    """
    Test the simulation maps.
    """
    frequencies = [1.42, 2.73, 5, 10, 30, 70, 143, 217, 353, 545, 857]

    w = WCS(naxis=2)
    w.wcs.crpix = [15, 15]
    w.wcs.cdelt = [1 / 60, 1 / 60]
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["GLON-CYP", "GLAT-CYP"]
    shape = np.array((29, 29))

    model = (
        continuum.synchrotron_1comp()
        + continuum.freeFree_7500k()
        + continuum.ame_lognormal()
        + continuum.thermalDust()
    )

    sim_maps = []

    for f in frequencies:
        m = m_sims.stokesMapSimulation()
        m.add_simulation_component(
            m_sims.pointsrc.pointSource(I=model(f, np.radians(5 / 60) ** 2), fwhm_deg=5 / 60),
        )
        m.set_projection((w, shape))
        sim_maps.append(m)

    assert True
