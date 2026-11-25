import warnings

import numpy as np


def _powerlaw_tail(f_edge, ratio, exponents):
    '''
    Extrapolate a power law past an array edge as f_edge * ratio**exponents.

    The naive expression yields 0/0 when the edge is zero and nan/inf when the
    ratio is not a positive finite number. Keep a zero edge at zero and fall
    back to constant padding in the pathological cases.
    '''
    if f_edge == 0:
        return np.zeros(len(exponents))
    if not np.isfinite(ratio) or ratio <= 0:
        return np.full(len(exponents), f_edge)
    return f_edge * ratio ** exponents


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
        except (TypeError, ValueError):
            raise TypeError(
                "Please enter valid x range in the form of a tuple (x_min, x_max) or list [x_min, x_max]."
            )
    else:
        x_min = None
        x_max = None

    try:
        ext_left, ext_right = ext
    except TypeError:
        ext_left = ext_right = ext

    for ext_ in (ext_left, ext_right):
        if ext_ not in (0, 1, 2, 3):
            raise ValueError(
                "Please provide valid values for ext argument (i.e. 0, 1, 2, 3), got %r." % (ext_,)
            )

    x, f, N_left, N_right = padding(
        x, f, ext_left=ext_left, ext_right=ext_right, n_ext=0
    )

    max_pass = 100 # each pass doubles the array; a distant range must fail, not OOM
    n_pass = 0

    if (x_min is not None) and (x_max is not None):

        if ext_left > 0 and ext_right > 0:
            while x[0] > x_min and x[-1] < x_max:
                n_pass += 1
                if n_pass > max_pass:
                    raise ValueError('Extending x to [%g, %g] needs more than %d passes.'
                                     % (x_min, x_max, max_pass))
                x, f, N_left_prime, N_right_prime = padding(
                    x, f, ext_left=ext_left, ext_right=ext_right, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if x_min is not None:

        if ext_left > 0:
            while x[0] > x_min and (x_max is None or x[-1] >= x_max):
                n_pass += 1
                if n_pass > max_pass:
                    raise ValueError('Extending x down to %g needs more than %d passes.'
                                     % (x_min, max_pass))
                x, f, N_left_prime, N_right_prime = padding(
                    x, f, ext_left=ext_left, ext_right=0, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if x_max is not None:

        if ext_right > 0:
            while x[-1] < x_max and (x_min is None or x[0] <= x_min):
                n_pass += 1
                if n_pass > max_pass:
                    raise ValueError('Extending x up to %g needs more than %d passes.'
                                     % (x_max, max_pass))
                x, f, N_left_prime, N_right_prime = padding(
                    x, f, ext_left=0, ext_right=ext_right, n_ext=1
                )
                N_left += N_left_prime
                N_right += N_right_prime

    if (x_min is not None and x[0] < x_min) or (x_max is not None and x[-1] > x_max):
        warnings.warn('The requested range is already inside the input x; '
                      'extrapolation was a no-op.', UserWarning)

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
                with np.errstate(divide='ignore', invalid='ignore'):
                    f_left = _powerlaw_tail(f[0], f[1] / f[0], np.arange(-N_left, 0))
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
                with np.errstate(divide='ignore', invalid='ignore'):
                    f_right = _powerlaw_tail(f[-1], f[-1] / f[-2], np.arange(1, N_right + 1))
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
