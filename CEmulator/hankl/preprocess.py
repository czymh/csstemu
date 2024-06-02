import numpy as np


def preprocess(x, f, ext=0, range=None):
    r"""
    This is the main preprocessing tool. It's purpose is to zero/constant pad or extrapolate the input arrays in a consistent way.

    Parameters
    ----------
	x : array
        Array of uniformly logarithmically spaced x values.
	f : array
        Array of respective F(x) values.
    ext : int or tuple or list
        Controls the extrapolation mode. When ext is an integer then the same extrapolation method will
        be used for both ends of the input array. Alternatively, when ext is an tuple (ext_left, ext_right) or a list [ext_left,
        ext_right] then different methods can be used for the two ends of the the input array.

        * if ext=0 then no extrapolation is performed (Default).
        * if ext=1 then zero padding is performed.
        * if ext=2 then constant padding is performed.
        * if ext=3 then Power-Law extrapolation is performed.

    range : tuple or list
        The minimum extrapolation range in the form of a tuple (x_min, x_max) or list [x_min, x_max]. When
        range=None (Default) then the extended range is chosen automatically such that its array-size is the next power of two.

    Returns
    -------
    x : array
        Extended range.
    f : array
        Extrapolated input array.
    N_left : int
        Number of array elements that were added to the left of the initial x range.
    N_right : int
        Number of array elements that were added to the right of the initial x range.
    """

    if range is not None:
        try:
            x_min, x_max = range
        except:
            raise TypeError(
                "Please enter valid x range in the form of a tuple (x_min, x_max) or list [x_min, x_max]."
            )
    else:
        x_min = None
        x_max = None

    try:
        ext_left, ext_right = ext
    except:
        ext_left = ext_right = ext

    x, f, N_left, N_right = padding(
        x, f, ext_left=ext_left, ext_right=ext_right, n_ext=0
    )

    if (x_min is not None) and (x_max is not None):

        if ext_left > 0 and ext_right > 0:
            while x[0] > x_min and x[-1] < x_max:
                x, f, N_left_prime, N_right_prime = padding(
                    x, f, ext_left=ext_left, ext_right=ext_right, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if x_min is not None:

        if ext_left > 0:
            while x[0] > x_min and (x_max is None or x[-1] >= x_max):
                x, f, N_left_prime, N_right_prime = padding(
                    x, f, ext_left=ext_left, ext_right=0, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if x_max is not None:

        if ext_right > 0:
            while x[-1] < x_max and (x_min is None or x[0] <= x_min):
                x, f, N_left_prime, N_right_prime = padding(
                    x, f, ext_left=0, ext_right=ext_right, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    return x, f, N_left, N_right


def padding(x, f, ext_left=0, ext_right=0, n_ext=0):
    r"""
    This function extends the input arrays until they reach the next-power-of-two size array.

    Parameters
    ----------
	x : array
        Array of uniformly logarithmically spaced x values.
	f : array
        Array of respective F(x) values.
    ext_left : int
        Controls the extrapolation mode for the left segment:

        * if ext=0 then no extrapolation is performed (Default).
        * if ext=1 then zero padding is performed.
        * if ext=2 then constant padding is performed.
        * if ext=3 then Power-Law extrapolation is performed.

    ext_right : int
        Controls the extrapolation mode for the right segment:

        * if ext=0 then no extrapolation is performed (Default).
        * if ext=1 then zero padding is performed.
        * if ext=2 then constant padding is performed.
        * if ext=3 then Power-Law extrapolation is performed.

    n_ext : int
        When n_ext=0 (Default) then the input array is extended until it reaches the next power of two size.
        When n_ext=1 then the input array is extended until is reaches the next power of two but one and so on. 

    Returns
    -------
    x : array
        Extended range.
    f : array
        Extrapolated input array.
    N_left : int
        Number of array elements that were added to the left of the initial x range.
    N_right : int
        Number of array elements that were added to the right of the initial x range.
    """

    N = x.size
    if N < 2:
        raise ValueError("Size of input arrays needs to be larger than 2")
    N_prime = 2 ** ((N - 1).bit_length() + n_ext)

    if N_prime > N:
        N_tails = N_prime - N

        if ext_left > 0 and ext_right > 0:
            N_left = N_tails // 2
            N_right = N_tails - N_left
        elif ext_left > 0 and ext_right < 1:
            N_left = N_tails
            N_right = 0
        elif ext_left < 1 and ext_right > 0:
            N_left = 0
            N_right = N_tails
        elif ext_left < 1 and ext_right < 1:
            return x, f, 0, 0
        else:
            raise ValueError(
                "Please provide valid values for ext argument (i.e. 0, 1, 2, 3)"
            )

        delta = (np.log10(np.max(x)) - np.log10(np.min(x))) / float(N - 1)
        x_prime = np.logspace(
            np.log10(x[0]) - N_left * delta, np.log10(x[-1]) + N_right * delta, N_prime
        )

        if N_left > 0:
            if ext_left == 1:
                f_left = np.zeros(N_left)
            elif ext_left == 2:
                f_left = np.full(N_left, f[0])
            elif ext_left == 3:
                f_left = f[0] * (f[1] / f[0]) ** np.arange(-N_left, 0)
            else:
                raise ValueError(
                    "Please provide valid values for ext argument (i.e. 0, 1, 2, 3)"
                )
        else:
            f_left = np.array([])

        if N_right > 0:
            if ext_right == 1:
                f_right = np.zeros(N_right)
            elif ext_right == 2:
                f_right = np.full(N_right, f[-1])
            elif ext_right == 3:
                f_right = f[-1] * (f[-1] / f[-2]) ** np.arange(1, N_right + 1)
            else:
                raise ValueError(
                    "Please provide valid values for ext argument (i.e. 0, 1, 2, 3)"
                )
        else:
            f_right = np.array([])

        f_prime = np.nan_to_num(np.concatenate((f_left, f, f_right)))

        return x_prime, f_prime, N_left, N_right

    else:
        return x, f, 0, 0
