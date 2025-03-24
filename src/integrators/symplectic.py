from numpy._typing import NDArray

from integrators.gradient import force_calc


class Symplectic:
    def __init__(self, order):
        self.order = order
        self.cc = None
        self.dd = None
        self.nsym = None

        # Get symplectic dimension (nsym) and position and momentum coeffs (cc, dd):
        self.initialize_symplectic()

    def __call__(
        self,
        qcinput: dict,
        dt: float,
        wmass: NDArray,
        q: NDArray,
        p: NDArray,
        atoms: list[str],
    ) -> tuple[NDArray, NDArray]:
        """
        Symplectic integrator for 4th, 6th, or 8th order integration.
        Use order to choose which one you need

        Ref 4th order: Qin, M., Wang, D., Zhang, M. J. Comp. Math.,1991, 9: pp211—221
        Ref 6th and 8th: H. Yoshida, Physics Letters A. 1990, 150, pp 262.
        """
        for j in range(self.nsym):
            q += self.cc[j] * p / wmass * dt
            force = force_calc(qcinput, q, atoms)
            p += self.dd[j] * force * dt

        # Final update for position with the last symplectic coefficient
        q += self.cc[self.nsym] * p / wmass * dt  # cc(nsym) in Fortran

        return q, p

    def initialize_symplectic(self) -> None:
        """
        Initialize symplectic integration coefficients based on the specified order.

        Parameters:
            order (int): The order of integration (4, 6, or 8).

        Returns:
            nsym (int): The number of symplectic stages.
            dd: Array of force coefficients (used for p-update)
            cc: Array of position coefficients (used for q-update)
        """
        if self.order == 4:
            # Reference for 4th order: Qin, M., Wang, D., Zhang, M. J. Comp. Math., 1991, 9: pp211—221
            ax = 2 ** (1 / 3) + (0.5) ** (1 / 3)

            self.dd = [(2.0 + ax) / 3.0, -(1.0 + 2.0 * ax) / 3.0, (2.0 + ax) / 3.0]

            self.cc = [
                (2.0 + ax) / 6.0,
                (1.0 - ax) / 6.0,
                (1.0 - ax) / 6.0,
                (2.0 + ax) / 6.0,
            ]

            self.nsym = 3

        elif self.order == 6:
            # Reference for 6th order: H. Yoshida, Physics Letters A. 1990, 150, pp 262.
            # A-solution from Yoshida:
            w1 = -1.17767998417887
            w2 = 0.235573213359357
            w3 = 0.784513610477560
            w4 = 1.0 - 2.0 * (w1 + w2 + w3)
            # B-solution from Yoshida:
            # w1  = -0.213228522200144e1
            # w2  =  0.426068187079180e-2
            # w3  =  0.143984816797678e1
            # w4 =  1.0 - 2.0 * (w1+w2+w3)

            wi = [w1, w2, w3, w4]

            self.dd = [wi[2], wi[1], wi[0], wi[3], wi[0], wi[1], wi[2]]

            self.cc = [
                0.5 * wi[2],
                0.5 * (wi[2] + wi[1]),
                0.5 * (wi[1] + wi[0]),
                0.5 * (wi[0] + wi[3]),
                0.5 * (wi[0] + wi[3]),
                0.5 * (wi[1] + wi[0]),
                0.5 * (wi[2] + wi[1]),
                0.5 * wi[2],
            ]

            self.nsym = 7

        elif self.order == 8:
            # Reference for 8th order: H. Yoshida, Physics Letters A. 1990, 150, pp 262.
            # Solution D coefficients
            w1 = 0.102799849391985
            w2 = -1.96061023297549
            w3 = 1.93813913762276
            w4 = -0.158240635368243
            w5 = -1.44485223686048
            w6 = 0.253693336566229
            w7 = 0.914844246229740
            w8 = 1.0 - 2.0 * (w1 + w2 + w3 + w4 + w5 + w6 + w7)
            # A-solution from Yoshida:
            # w1  = -0.161582374150097e1
            # w2  = -0.244699182370524e1
            # w3  = -0.716989419708120e-2
            # w4  =  0.244002732616735e1
            # w5  =  0.157739928123617e0
            # w6  =  0.182020630970714e1
            # w7  =  0.104242620869991e1
            # w8  =  1.0 - 2.0*(w1 + w2 + w3 + w4 + w5 + w6 + w7)

            wi = [w1, w2, w3, w4, w5, w6, w7, w8]

            self.dd = [
                wi[6],
                wi[5],
                wi[4],
                wi[3],
                wi[2],
                wi[1],
                wi[0],
                wi[7],
                wi[0],
                wi[1],
                wi[2],
                wi[3],
                wi[4],
                wi[5],
                wi[6],
            ]

            self.cc = [
                0.5 * wi[6],
                0.5 * (wi[6] + wi[5]),
                0.5 * (wi[5] + wi[4]),
                0.5 * (wi[4] + wi[3]),
                0.5 * (wi[3] + wi[2]),
                0.5 * (wi[2] + wi[1]),
                0.5 * (wi[1] + wi[0]),
                0.5 * (wi[0] + wi[7]),
                0.5 * (wi[0] + wi[7]),
                0.5 * (wi[1] + wi[0]),
                0.5 * (wi[2] + wi[1]),
                0.5 * (wi[3] + wi[2]),
                0.5 * (wi[4] + wi[3]),
                0.5 * (wi[5] + wi[4]),
                0.5 * (wi[6] + wi[5]),
                0.5 * wi[6],
            ]

            self.nsym = 15

        else:
            raise ValueError(
                "Wrong order in Symplectic! Only order = 4,6,8 is avialable"
            )
