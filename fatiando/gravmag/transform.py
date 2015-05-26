"""
Space domain potential field transformations, like upward continuation,
derivatives and total mass.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of the
  vertical component of gravity :math:`g_z` using numerical integration
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)
* :func:`~fatiando.gravmag.transform.mstde`: Multi-scale tilt depth estimation
  method.
* :func:`~fatiando.gravmag.transform.tilt`: Calculates the tilt angle
* :func:`~fatiando.gravmag.transform.tdx`: Calculates the horizontal tilt angle

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction
* :func:`~fatiando.gravmag.transform.thdr`: Utility function to generate the
  total horizontal derivative

----

"""
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


def tga(x, y, data, shape):
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
    * shape : tuple = (ny, nx)
        The shape of the grid

    Returns:

    * tga : 1D-array
        The amplitude of the total gradient

    """
    dx = derivx(x, y, data, shape)
    dy = derivy(x, y, data, shape)
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

    return tiltd


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

def derivx(x, y, data, shape, order=1):
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
    * shape : tuple = (ny, nx)
        The shape of the grid
    * order : int
        The order of the derivative

    Returns:

    * deriv : 1D-array
        The derivative

    """
    Fx = _getfreqs(x, y, data, shape)[0].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fx * 1j, data, shape, order)


def derivy(x, y, data, shape, order=1):
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
    * shape : tuple = (ny, nx)
        The shape of the grid
    * order : int
        The order of the derivative

    Returns:

    * deriv : 1D-array
        The derivative

    """
    Fy = _getfreqs(x, y, data, shape)[1].astype('complex')
    # Multiply by 1j because I don't multiply it in _deriv (this way _deriv can
    # be used for the z derivative as well)
    return _deriv(Fy * 1j, data, shape, order)


def derivz(x, y, data, shape, order=1):
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
    * shape : tuple = (ny, nx)
        The shape of the grid
    * order : int
        The order of the derivative

    Returns:

    * deriv : 1D-array
        The derivative

    """
    Fx, Fy = _getfreqs(x, y, data, shape)
    freqs = numpy.sqrt(Fx ** 2 + Fy ** 2)
    return _deriv(freqs, data, shape, order)


def thdr(x, y, data, shape):
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


def _getfreqs(x, y, data, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    ny, nx = shape
    dx = float(x.max() - x.min()) / float(nx - 1)
    fx = numpy.fft.fftfreq(nx, dx)
    dy = float(y.max() - y.min()) / float(ny - 1)
    fy = numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fx, fy)


def _deriv(freqs, data, shape, order):
    """
    Calculate a generic derivative using the FFT.
    """
    fgrid = (2. * numpy.pi) * numpy.fft.fft2(numpy.reshape(data, shape))
    deriv = numpy.real(numpy.fft.ifft2((freqs ** order) * fgrid).ravel())
    return deriv
