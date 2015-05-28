"""
Potential field transformations, like upward continuation and derivatives.

.. note:: Most, if not all, functions here required gridded data.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of
  gridded potential field data on a level surface
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)
* :func:`~fatiando.gravmag.transform.mstde`: Multi-scale tilt depth estimation
  method.
* :func:`~fatiando.gravmag.transform.tilt`: Calculates the tilt angle
* :func:`~fatiando.gravmag.transform.tdx`: Calculates the horizontal tilt angle

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction (North-South)
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction (East-West)
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction
* :func:`~fatiando.gravmag.transform.thdr`: Utility function to generate the
  total horizontal derivative

----

"""
from __future__ import division
import numpy
from math import atan,sqrt


def upcontinue(gz, height, xp, yp, dims):
    """
    Upward continue :math:`g_z` data using numerical integration of the
    analytical formula:

    .. math::

        g_z(x,y,z) = \\frac{z-z_0}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^
        {\infty} g_z(x',y',z_0) \\frac{1}{[(x-x')^2 + (y-y')^2 + (z-z_0)^2
        ]^{\\frac{3}{2}}} dx' dy'

    .. note:: Data needs to be on a regular grid!

    .. note:: Units are SI for all coordinates and mGal for :math:`g_z`

    .. note:: be aware of coordinate systems!
        The *x*, *y*, *z* coordinates are:
        x -> North, y -> East and z -> **DOWN**.

    Parameters:

    * gz : array
        The gravity values on the grid points
    * height : float
        How much higher to move the gravity field (should be POSITIVE!)
    * xp, yp : arrays
        The x and y coordinates of the grid points
    * dims : list = [dy, dx]
        The grid spacing in the y and x directions

    Returns:

    * gzcont : array
        The upward continued :math:`g_z`

    """
    if xp.shape != yp.shape:
        raise ValueError("xp and yp arrays must have same shape")
    if height < 0:
        raise ValueError("'height' should be positive")
    dy, dx = dims
    area = dx * dy
    deltaz_sqr = (height) ** 2
    gzcont = numpy.zeros_like(gz)
    for x, y, g in zip(xp, yp, gz):
        gzcont += g * area * \
            ((xp - x) ** 2 + (yp - y) ** 2 + deltaz_sqr) ** (-1.5)
    gzcont *= abs(height) / (2 * numpy.pi)
    return gzcont


def tga(x, y, data, shape, method='fd'):
    """
    Calculate the total gradient amplitude.

    This the same as the `analytic signal`, but we prefer the newer, more
    descriptive nomenclature.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units and so will the total gradient amplitude! I strongly
        recommend converting the data to SI **before** calculating the
        derivative (use one of the unit conversion functions of
        :mod:`fatiando.utils`).

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * method : string
        The method used to calculate the horizontal derivatives. Options are:
        ``'fd'`` for finite-difference (more stable) or ``'fft'`` for the Fast
        Fourier Transform. The z derivative is always calculated by FFT.

    Returns:

    * tga : 1D-array
        The amplitude of the total gradient

    """
    dx = derivx(x, y, data, shape, method=method)
    dy = derivy(x, y, data, shape, method=method)
    dz = derivz(x, y, data, shape)
    res = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return res


def mstde(x, y, data, shape):
    """
    Multi-Scale Tilt Depth Estimation

    Implements the method of van Buren (2013).

    An empirical method for attempting to approximate the depth of the source
    of a magnetic anomaly, based on altering tilt and continuation of the
    magnetic field intensity.

    This method will also work well on Total Gradient Amplitude, but care must
    be taken to reduce noise.

    .. warning::

        If the data is not in SI units, the derivatives will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivatives (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    **References**

    Van Buren, Reece. 2013. “Multi-Scale Tilt Depth Estimation.” MSc Thesis,
    Johannesberg: University of the Witwatersrand.
    URI: http://mobile.wiredspace.wits.ac.za/bitstream/handle/10539/14011/MSc_Dissertation_RvanBuren_2013.pdf.
    """

    tilted_data = tilt(x, y, data, shape)
    pass


def tilt(x, y, data, shape):
    """
    Calculates the magnetic tilt, as defined by Miller and Singh (1994):

    tilt(f) = tan^{-1}(\\frac{\\frac{df}{dz}}{\\sqrt{\\frac{df}{dx}^2 + \\frac{df}{dy}^2}})

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * tilt : 1D-array
        The tilt angle of the total field.

    """
    horiz_deriv = thdr(x, y, data, shape)
    vert_deriv = derivz(x, y, data, shape)
    tilt_value = vert_deriv/horiz_deriv
    tilt = numpy.arctan2( tilt_value, tilt_value )

    return tilt


'''def tdx(x, y, data, shape):
    """
    Horizontal tilt angle, from Cooper and Cowan (2006):

    tilt(f) = tan^{-1}(\\frac{\\frac{df}{dz}}{|total_horizonal_derivative|})

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * tilt : 1D-array
        The tilt angle of the total field.

    """
    horiz_deriv = thdr(x, y, data, shape)
    abs_vert_deriv = numpy.absolute(derivz(x, y, data, shape))
    tilt_value = horiz_deriv/abs_vert_deriv
    tdx_value = numpy.arctan2( tilt_value, abs_vert_deriv )

    return tdx_value
'''


def derivx(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the x direction.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        kx, _ = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(kx*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx : padx + nx, pady : pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dx = (x.max() - x.min())/(nx - 1)
        deriv = numpy.empty_like(datamat)
        deriv[1:-1, :] = (datamat[2:, :] - datamat[:-2, :])/(2*dx)
        deriv[0, :] = deriv[1, :]
        deriv[-1, :] = deriv[-2, :]
        if order > 1:
            deriv = derivx(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivy(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the y direction.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        _, ky = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(ky*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx : padx + nx, pady : pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dy = (y.max() - y.min())/(ny - 1)
        deriv = numpy.empty_like(datamat)
        deriv[:, 1:-1] = (datamat[:, 2:] - datamat[:, :-2])/(2*dy)
        deriv[:, 0] = deriv[:, 1]
        deriv[:, -1] = deriv[:, -2]
        if order > 1:
            deriv = derivy(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivz(x, y, data, shape, order=1, method='fft'):
    """
    Calculate the derivative of a potential field in the z direction.

    .. warning::

        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fft'`` for the Fast Fourier Transform.

    Returns:

    * deriv : 1D-array
        The derivative

    """
    assert method == 'fft', \
        "Invalid method '{}'".format(method)
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    deriv_ft = numpy.fft.fft2(padded)*numpy.sqrt(kx**2 + ky**2)**order
    deriv = numpy.real(numpy.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx : padx + nx, pady : pady + ny].ravel()


def _pad_data(data, shape):
    n = _nextpow2(numpy.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = numpy.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
    return padded, padx, pady


def _nextpow2(i):
    buf = numpy.ceil(numpy.log(i)/numpy.log(2))
    return int(2**buf)


def _fftfreqs(x, y, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(padshape[1], dy)
    return numpy.meshgrid(fy, fx)[::-1]


def thdr(x, y, data, shape, method = 'fd'):
    """
    Total Horizontal Derivative

    A useful thing used in many tilt angle filters:

    sqrt( \\frac{df}{dx}^2 + \\frac{df}{dy}^2 )

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * tilt : 1D-array
        The tilt angle of the total field.

    """
    total_horiz_deriv = numpy.sqrt( derivx(x, y, data, shape)**2 + derivy(x, y, data, shape)**2 )

    return total_horiz_deriv
