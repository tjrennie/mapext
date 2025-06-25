import astropy.units as astropy_u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.wcs.utils import skycoord_to_pixel
from regions import CircleSkyRegion, CircleAnnulusSkyRegion
from mapext.core.stokes import display_parameters

def apPhoto(astroMap, foreground, background):
    """Perform aperture photometry on a single AstroMap object using a foreground region and background region.
    
    Parameters
    ----------
    astroMap : AstroMap
        The AstroMap object containing the data to analyze.
    foreground : SkyRegion
        The region defining the source for which to perform photometry.
    background : SkyRegion
        The region defining the background for which to perform photometry.
    
    Returns
    -------
    Sv : list
        List of measured fluxes for each Stokes parameter.
    Sv_e : list
        List of uncertainties in the measured fluxes for each Stokes parameter.
    Sv_stokes : list
        List of Stokes parameters considered in the analysis.
    """
    src_mask = foreground.to_pixel(astroMap.projection).to_mask(mode='center').to_image(astroMap.shape)
    bkg_mask = background.to_pixel(astroMap.projection).to_mask(mode='center').to_image(astroMap.shape)

    astroMap.I

    Sv_stokes = astroMap._maps_cached
    Sv = []
    Sv_e = []
    if astroMap.assume_v_0:
        Sv_stokes = [stokes for stokes in Sv_stokes if stokes != 'V']

    for stokes in Sv_stokes:
        compmap = getattr(astroMap, stokes)

        src_sum = np.nansum(compmap * src_mask)
        src_cnt = np.nansum(src_mask)

        bkg_sum = np.nansum(compmap * bkg_mask)
        bkg_std = np.nanstd(compmap[bkg_mask>0.5])
        bkg_cnt = np.nansum(bkg_mask)
        
        Sv.append(src_sum - (bkg_sum / bkg_cnt) * src_cnt)
        Sv_e.append(bkg_std * np.sqrt(src_cnt * (1+ (np.pi/2)*(src_cnt/bkg_cnt))))

    return Sv, Sv_e, Sv_stokes

def bullseye(astroMap, astroSrc, aperture=5/60, annulus=[10/60,15/60]):

    region_src = CircleSkyRegion(
        center=astroSrc.coord,
        radius=aperture * astropy_u.deg
    )
    region_bkg = CircleAnnulusSkyRegion(
        center=astroSrc.coord,
        inner_radius=annulus[0] * astropy_u.deg,
        outer_radius=annulus[1] * astropy_u.deg
    )

    bullseye_plot(astroMap, astroSrc, region_src, region_bkg)

    # Sv, Sve, Svs = apPhoto(astroMap, foreground=region_src, background=region_bkg)

    print(Sv, Sve, Svs)

def bullseye_plot(astroMap, astroSrc, aperture_region, annulus_region,
                  components='core', assume_v_0=True):
    """Plot the bullseye region for aperture photometry.
    
    Parameters
    ----------
    astroMap : AstroMap
        The AstroMap object containing the data to analyze.
    astroSrc : AstroSource
        The AstroSource object containing the source coordinates.
    foreground_region : SkyRegion
        The region defining the source for which to perform photometry.
    background_region : SkyRegion
        The region defining the background for which to perform photometry.
    components : str or list, optional
        The components of the Stokes parameters to plot. If 'all', all components are plotted
        (default is 'all').
    """
    if components == 'all':
        components = list(display_parameters.keys())
        if assume_v_0:
            components = [comp for comp in components if comp != 'V']
            rows = [components[:-3], components[-3:]]
        else:
            rows = [components[:1], components[1:-3], components[-3:]]
    if components == 'core':
        components = ['I', 'Q', 'U', 'V']
        if assume_v_0:
            components.remove('V')
        rows = [components]
    else:
        rows = [components]

    if len(rows) > 1:
        gridshape = (len(rows), 2*max(*[len(x) for x in rows]))
    else:
        gridshape = (1, 2*len(rows[0]))

    fig = plt.figure(figsize=(gridshape[1]*3/2, gridshape[0]*4))
    gs = gridspec.GridSpec(gridshape[0], gridshape[1], figure=fig, wspace=0.2, hspace=0.2)
    axs = []

    for i, row in enumerate(rows):
        row_len = len(row)
        for j, comp in enumerate(row):
            # Create axis
            axs.append(fig.add_subplot(gs[i, gridshape[1]//2 - row_len + 2*j : gridshape[1]//2 - row_len + 2*j + 2], projection=astroMap.projection, sharex=axs[0] if i > 0 else None, sharey=axs[0] if j > 0 else None))
            # Obtain maps
            try:
                m = getattr(astroMap, comp)
            except ValueError:
                m = np.full(astroMap.shape, 0)
            if comp == 'A':
                m = np.degrees(m)  # Convert radians to degrees for the angle map
            # Plot
            # imshow
            if comp =='A':
                im_base = axs[-1].imshow(m, origin='lower', vmin=0, vmax=180)
            else:
                im_base = axs[-1].imshow(m, origin='lower')
            aperture_region.to_pixel(astroMap.projection).plot(ax=axs[-1], color='w', lw=1, ls='dashed', label='Aperture')
            annulus_region.to_pixel(astroMap.projection).plot(ax=axs[-1], color='w', lw=1, ls='dashed', label='Annulus')
            x, y = skycoord_to_pixel(astroSrc.coord, astroMap.projection)
            axs[-1].plot(x, y, 'x', color='green', label='Source Position')
            # Labels and titles
            axs[-1].set_title(f"{comp}")
            axs[-1].set_xlabel(astroMap.projection.wcs.ctype[0])
            axs[-1].set_ylabel(astroMap.projection.wcs.ctype[1])
            # define and label units
            if comp in ['I','Q','U','V','P']:
                units = 'Jy/beam' #astroMap.unit.to_string()
            elif comp == 'A':
                units = 'Degrees'
            elif comp == 'PF':
                units = 'N/A'
            else:
                raise ValueError(f"Unknown component {comp} for units.")
            plt.colorbar(im_base, ax=axs[-1], orientation='horizontal', label=units)

    plt.show()
