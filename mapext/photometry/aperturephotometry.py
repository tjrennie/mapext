"""Module providing functions for performing aperture photometry on astronomical maps, including plotting utilities for visualizing aperture and annulus regions."""

import logging

import astropy.units as astropy_u
import colorcet as cc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs.utils import skycoord_to_pixel
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from mapext.core.stokes import display_parameters

logger = logging.getLogger(__name__)


def apPhoto(astro_map, foreground, background):
    """Perform aperture photometry on a single AstroMap object using a foreground region and background region.

    Parameters
    ----------
    astro_map : AstroMap
        The AstroMap object containing Stokes parameters.
    foreground : CircleSkyRegion
        The region defining the source aperture.
    background : CircleAnnulusSkyRegion
        The region defining the background annulus.
    """
    src_mask = (
        foreground.to_pixel(astro_map.projection)
        .to_mask(mode="center")
        .to_image(astro_map.shape)
    )
    bkg_mask = (
        background.to_pixel(astro_map.projection)
        .to_mask(mode="center")
        .to_image(astro_map.shape)
    )

    if (src_mask is None) or (bkg_mask is None):
        logger.warning(
            "No pixels in the source or background region. Returning empty results."
        )
        return [], [], []

    Sv_stokes = astro_map._maps_cached
    Sv = []
    Sv_e = []
    if astro_map.assume_v_0:
        Sv_stokes = [stokes for stokes in Sv_stokes if stokes != "V"]

    for stokes in Sv_stokes:
        compmap = getattr(astro_map, stokes)

        src_sum = np.nansum(compmap * src_mask)
        src_cnt = np.nansum(src_mask * np.isfinite(compmap))

        bkg_med = np.nanmedian(compmap[bkg_mask > 0.5])
        bkg_std = np.nanstd(compmap[bkg_mask > 0.5])
        bkg_cnt = np.nansum((bkg_mask > 0.5) * np.isfinite(compmap))

        calibration_frac = 0.01 * astro_map._calibration.get(stokes, {}).get(
            "percentage", 0
        )

        S_nu = src_sum - (bkg_med * src_cnt)
        Sv.append(S_nu)

        bkg_err_sq = bkg_std**2 * src_cnt * (1 + (np.pi / 2) * (src_cnt / bkg_cnt))
        cal_err_sq = (calibration_frac * S_nu) ** 2

        S_nu_err = np.sqrt(bkg_err_sq + cal_err_sq)
        Sv_e.append(S_nu_err)

    return Sv, Sv_e, Sv_stokes


def apertureAnnulus(
    astro_map,
    astro_source,
    aperture=5 / 60,
    annulus=[10 / 60, 15 / 60],
    plot=False,
    components="core",
    assume_v_0=True,
    save_path=None,
    return_results=False,
    verbose=True,
    result_to_src=True,
):
    """Perform aperture photometry and optionally plot the aperture layout.

    Parameters
    ----------
    astro_map : AstroMap
        Map object containing Stokes parameters.
    astro_source : AstroSource
        Source object containing sky coordinates.
    aperture : float
        Radius of the circular aperture in degrees.
    annulus : list of float
        Inner and outer radius of the background annulus in degrees.
    plot : bool
        Whether to generate the bullseye plot.
    components : str or list
        Which Stokes components to plot.
    assume_v_0 : bool
        Whether to assume Stokes V is zero and omit from analysis.
    save_path : str or None
        File path to save the plot. If None, the plot is not saved.
    return_results : bool
        If True, return the photometry results.
    verbose : bool
        If True, print photometry results.
    result_to_src : bool
        If True, store the results in the astro_source object.

    Returns
    -------
    If return_results is True:
        tuple: (Sv, Sve, Svs)
    """
    region_src = CircleSkyRegion(
        center=astro_source.coord, radius=aperture * astropy_u.deg
    )
    region_bkg = CircleAnnulusSkyRegion(
        center=astro_source.coord,
        inner_radius=annulus[0] * astropy_u.deg,
        outer_radius=annulus[1] * astropy_u.deg,
    )

    Sv, Sve, Svs = apPhoto(astro_map, foreground=region_src, background=region_bkg)

    if len(Sv) == 0 and len(Sve) == 0 and len(Svs) == 0:
        logger.warning(
            "No pixels in the source or background region. Returning empty results."
        )
        return [], [], []

    if verbose:
        for s, f, e in zip(Svs, Sv, Sve):
            print(f"{s}: {f:.4g} ± {e:.4g}")

    if plot!=False:
        fig = apPhoto_regionPlot(
            astro_map,
            astro_source,
            region_src,
            region_bkg,
            components="core" if plot is True else plot,
            assume_v_0=assume_v_0,
        )
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            if verbose:
                print(f"Plot saved to: {save_path}")
        if not save_path:
            plt.show()
        plt.close(fig)

    if result_to_src:
        values = dict(zip(Svs, Sv))
        errors = dict(zip(Svs, Sve))
        astro_source.add_flux(
            name=astro_map.name if hasattr(astro_map, "name") else "none",
            freq=astro_map.frequency.value if hasattr(astro_map, "frequency") else 0,
            bandwidth=(
                astro_map.bandwidth.value if hasattr(astro_map, "bandwidth") else 0
            ),
            values=values,
            errors=errors,
            epoch=astro_map.epoch if hasattr(astro_map, "epoch") else None,
        )

    if return_results:
        return Sv, Sve, Svs, fig if plot else None
    return None


def apPhoto_regionPlot(
    astro_map,
    astro_source,
    aperture_region,
    annulus_region,
    components="all",
    assume_v_0=True,
):
    """Plot the bullseye region for aperture photometry.

    Parameters
    ----------
    astro_map : AstroMap
        Map object containing Stokes parameters.
    astro_source : AstroSource
        Source object containing sky coordinates.
    aperture_region : CircleSkyRegion
        Region defining the circular aperture.
    annulus_region : CircleAnnulusSkyRegion
        Region defining the annulus for background subtraction.
    components : str or list
        Which Stokes components to plot. Options are 'all', 'core', or a list
        of specific components.
    assume_v_0 : bool
        Whether to assume Stokes V is zero and omit it from the analysis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the bullseye plot.
    """
    if components == "all":
        components = list(display_parameters.keys())
        if assume_v_0:
            components = [comp for comp in components if comp != "V"]
            rows = [components[:-3], components[-3:]]
        else:
            rows = [components[:1], components[1:-3], components[-3:]]
    elif components == "core":
        components = ["I", "Q", "U", "V"]
        if assume_v_0:
            components.remove("V")
        rows = [components]
    else:
        rows = [components]

    if len(rows) > 1:
        gridshape = (len(rows), 2 * max(*[len(x) for x in rows]))
    else:
        gridshape = (1, 2 * len(rows[0]))

    fig = plt.figure(figsize=(gridshape[1] * 3 / 2, gridshape[0] * 4))
    gs = gridspec.GridSpec(
        gridshape[0], gridshape[1], figure=fig, wspace=0.2, hspace=0.2
    )
    axs = []

    for i, row in enumerate(rows):
        row_len = len(row)
        for j, comp in enumerate(row):
            axs.append(
                fig.add_subplot(
                    gs[
                        i,
                        gridshape[1] // 2
                        - row_len
                        + 2 * j : gridshape[1] // 2
                        - row_len
                        + 2 * j
                        + 2,
                    ],
                    projection=astro_map.projection,
                    sharex=axs[0] if i > 0 else None,
                    sharey=axs[0] if j > 0 else None,
                )
            )

            try:
                m = getattr(astro_map, comp)
            except (AttributeError, ValueError):
                m = np.full(astro_map.shape, np.nan)

            if comp == "A":
                m = np.degrees(m)

            if comp == "A":
                im_base = axs[-1].imshow(
                    m,
                    origin="lower",
                    vmin=0,
                    vmax=180,
                    cmap=cc.cm["cyclic_mygbm_30_95_c78"],
                )
            else:
                im_base = axs[-1].imshow(
                    m, origin="lower", cmap=cc.cm["linear_bmy_10_95_c78"]
                )

            x, y = skycoord_to_pixel(astro_source.coord, astro_map.projection)
            axs[-1].axhline(y, color="#000000", lw=1, alpha=0.5)
            axs[-1].axvline(x, color="#000000", lw=1, alpha=0.5)

            aperture_region.to_pixel(astro_map.projection).plot(
                ax=axs[-1], color="#00ECE1", lw=1, ls="solid", label="Aperture"
            )
            annulus_region.to_pixel(astro_map.projection).plot(
                ax=axs[-1], color="#00E73A", lw=1, ls="solid", label="Annulus"
            )

            axs[-1].set_title(f"{comp}")
            axs[-1].set_xlabel(astro_map.projection.wcs.ctype[0])
            axs[-1].set_ylabel(astro_map.projection.wcs.ctype[1])

            if comp in ["I", "Q", "U", "V", "P"]:
                units = astro_map._unit
            elif comp == "A":
                units = "Degrees"
            elif comp == "PF":
                units = "N/A"
            else:
                raise ValueError(f"Unknown component {comp} for units.")

            plt.colorbar(im_base, ax=axs[-1], orientation="horizontal", label=units)

    return fig
