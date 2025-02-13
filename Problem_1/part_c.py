import numpy as np
from scipy.integrate import nquad

###############################################################################
# 1) Closed-form (analytic) moments for v ~ N(mu, Sigma).
###############################################################################

def moment_analytic(mu, Sigma, powers):
    """
    Compute the moment  < v_1^p1 v_2^p2 v_3^p3 >
    for a 3D Gaussian with mean=mu, covariance=Sigma,
    by direct expansion up to the orders we need.

    Parameters
    ----------
    mu    : array_like, shape (3,)
    Sigma : array_like, shape (3,3)
    powers: tuple (p1, p2, p3), the exponents of (v1, v2, v3).

    Returns
    -------
    M : float
        The analytic value of <v_1^p1 v_2^p2 v_3^p3>.
    """
    p1, p2, p3 = powers
    # For convenience:
    m1, m2, m3 = mu
    S11, S12, S13 = Sigma[0,0], Sigma[0,1], Sigma[0,2]
    S22, S23      = Sigma[1,1], Sigma[1,2]
    S33           = Sigma[2,2]

    # -- 1) First-order moments --
    if powers == (1,0,0):
        return m1
    if powers == (0,1,0):
        return m2
    if powers == (0,0,1):
        return m3

    # -- 2) Second-order moments (e.g. <v1 v2>) --
    if powers == (1,1,0):
        return S12 + m1*m2
    if powers == (0,1,1):
        return S23 + m2*m3
    if powers == (1,0,1):
        return S13 + m1*m3

    # -- 3) Third-order, e.g. <v1^2 v2> --
    # Expand: v1^2 v2 = (m1+Y1)^2 (m2+Y2) and use
    # <Y1^2 Y2> = 2 * S12 * etc. We do the standard expansions:
    # <v1^2 v2> = m1^2 m2 + 2 m1 S12 + m2 S11 (assuming zero-mean for Y).
    if powers == (2,1,0):  # <v1^2 v2>
        return (m1**2)*m2 + 2*m1*S12 + m2*S11
    if powers == (0,2,1):  # <v2^2 v3>
        return (m2**2)*m3 + 2*m2*S23 + m3*S22
    if powers == (1,0,2):  # <v1 v3^2>, if needed
        return (m1)*(m3**2) + 2*m3*S13 + m1*S33

    # -- 4) Fourth-order, e.g. <v1^2 v2^2> --
    # By expansion or Isserlis' theorem:
    # <v1^2 v2^2> =
    #   m1^2 m2^2
    # + m1^2 S22
    # + m2^2 S11
    # + 4 m1 m2 S12
    # + S11 S22
    # + 2 S12^2
    if powers == (2,2,0):  # <v1^2 v2^2>
        return (
            (m1**2)*(m2**2)
            + (m1**2)*S22
            + (m2**2)*S11
            + 4*m1*m2*S12
            + S11*S22
            + 2*(S12**2)
        )
    if powers == (0,2,2):  # <v2^2 v3^2>
        return (
            (m2**2)*(m3**2)
            + (m2**2)*S33
            + (m3**2)*S22
            + 4*m2*m3*S23
            + S22*S33
            + 2*(S23**2)
        )
    if powers == (2,0,2):  # <v1^2 v3^2>, if you want that as well
        return (
            (m1**2)*(m3**2)
            + (m1**2)*S33
            + (m3**2)*S11
            + 4*m1*m3*S13
            + S11*S33
            + 2*(S13**2)
        )

    # If other combinations are needed, they can be added similarly.
    raise NotImplementedError(
        f"moment_analytic() not implemented for powers={powers}."
    )

###############################################################################
# 2) Numerical moments via ratio of integrals
###############################################################################

def compute_moment_numeric(A, w, powers):
    """
    Computes <v_1^p1 v_2^p2 v_3^p3> numerically via:
      [ ∫ v^p e^{...} dv ] / [ ∫ e^{...} dv ].
    Uses nquad for the numerator.  The denominator is closed-form.

    Parameters
    ----------
    A : array_like (3x3), positive-definite
    w : array_like (3,), the linear shift
    powers : (p1, p2, p3) exponents for (v1, v2, v3)

    Returns
    -------
    numeric_value : float
        Numerically estimated moment.
    """
    A = np.array(A, dtype=float)
    w = np.array(w, dtype=float).ravel()
    # The dimension is 3 in this example.

    # 2a) Numerator = ∫ v_1^p1 v_2^p2 v_3^p3 e^{-1/2 v^T A v + v^T w} dv
    def numerator_integrand(*vargs):
        v = np.array(vargs)
        # powers (p1, p2, p3)
        v_factor = (v[0]**powers[0])*(v[1]**powers[1])*(v[2]**powers[2])
        exponent = -0.5*np.dot(v, A @ v) + np.dot(v, w)
        return v_factor * np.exp(exponent)

    # We'll use the known closed-form for the denominator:
    #  denominator = Z = sqrt((2π)^3 / det(A)) * exp(1/2 w^T A^-1 w).
    Ainv = np.linalg.inv(A)
    detA = np.linalg.det(A)
    denom = np.sqrt((2*np.pi)**3 / detA)* np.exp(0.5 * w @ Ainv @ w)

    # bounds for each dimension: (-∞, ∞)
    bounds = [(-np.inf, np.inf)]*3

    # "nquad" for the numerator
    num_val, err_est = nquad(numerator_integrand, bounds)

    return num_val / denom

###############################################################################
# 3) Putting it all together for the specific A, w, and the requested moments.
###############################################################################

if __name__ == "__main__":

    # The matrix A and vector w as given:
    A = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ], dtype=float)
    w = np.array([1, 2, 3], dtype=float)

    # Compute mu, Sigma for reference:
    Ainv = np.linalg.inv(A)
    mu   = Ainv @ w
    Sigma= Ainv

    # The list of moments we want to check:
    #  ( label, powers )
    moments_to_check = [
        ("<v1>"        , (1,0,0)),
        ("<v2>"        , (0,1,0)),
        ("<v3>"        , (0,0,1)),
        ("<v1 v2>"     , (1,1,0)),
        ("<v2 v3>"     , (0,1,1)),
        ("<v1 v3>"     , (1,0,1)),
        ("<v1^2 v2>"   , (2,1,0)),
        ("<v2^2 v3>"   , (0,2,1)),
        ("<v1^2 v2^2>", (2,2,0)),
        ("<v2^2 v3^2>", (0,2,2))
    ]

    # Print a header
    print("Matrix A:\n", A)
    print("w =", w)
    print("\nmu   =", mu)
    print("Sigma=\n", Sigma)
    print()

    # Loop over each moment, do numeric vs analytic:
    for label, powers in moments_to_check:
        M_ana = moment_analytic(mu, Sigma, powers)
        M_num = compute_moment_numeric(A, w, powers)
        print(f"{label:10s} = analytic: {M_ana: .6g} | numeric: {M_num: .6g}")

