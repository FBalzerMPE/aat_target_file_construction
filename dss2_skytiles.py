

import astropy.units as u
import pylab as py
from astroplan import FixedTarget
from astroplan.plots import plot_finder_image
from astropy.coordinates import SkyCoord


#########################################
# quick way to plot skies:
#
def check_skies(infile, interactive=True):

    # read input file (in format from configure):
    lis = open(infile, 'r')
    lislines = lis.readlines()

    nlines = len(lislines)
    print('Number of lines in file: ', nlines)

    names = []
    good = []
    n = 0
    for line in lislines:

        py.clf()
        tmp = line.split()
        names.append('_'.join(tmp[2:3]))

        coordstr = ' '.join(tmp[4:10])
        print(coordstr)

        coord = SkyCoord(coordstr, unit=(u.hourangle, u.deg))
        target = FixedTarget(coord=coord, name=names[n])
        print(coord)
        print(target)
        ax, hdu = plot_finder_image(
            target, survey='DSS2 Red', fov_radius=1.0 * u.arcmin, reticle=True)
        #ax, hdu = plot_finder_image(target,survey='DSS',fov_radius=1.0*u.arcmin,reticle=True)
        py.show()
        print(tmp[0], n, ' of ', nlines)
        if (interactive):
            yn = input('Is this sky good (Y/N)')

            if ((yn == 'Y') | (yn == 'y')):
                good.append(1)
            else:
                good.append(0)

        n = n + 1

    if (interactive):
        print('all skies:')
        print(len(names), len(good), n)
        for i in range(n):
            print(names[i], good[i])

        print('bad skies:')
        for i in range(n):
            if (good[i] == 0):
                print(names[i], good[i])

    return
