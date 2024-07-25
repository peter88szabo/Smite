import numpy as np;
import random
import math


def diatom_init_harm(req, omega, redmass, jrot, nvib):
    q = []
    p = []

    dr_angle = random.uniform(0, 2 * math.pi)

    dr = math.sqrt( 2 * ( n + 0.5 ) / ( redmass * omega)) * math.cos(dr_angle)

    pr_angle = random.uniform(0, 2 * math.pi)
    pr = - math.sqrt(2 * ( n + 0.5 ) * redmass * omega) * math.sin(pr_angle)
     
    r = r_eq + dr

    q1 = [r, 0.0, 0.0]
    q2 = [0.0, 0.0, 0.0]

    vr = pr / redmass

    g = w[1] / sum(w)

    p1 = [w[0] * vr * g, 0.0, 0.0]
    p2 = [w[1] * (vr - p1[0]/w[0]), 0.0, 0.0]

    qq, pp = cenmass(q1+q2, p1+p2, w)

    agnmomabs = math.sqrt(jrot * ( jrot + 1))

    ai = [1.0e20, redmass * r * r, redmass * r * r]

    angle = random.uniform(0, 2 * math.pi)

    angmom = [
        0.0,
        agnmomabs * math.sin(angle),
        agnmomabs * math.cos(angle)
    ]

    wx = 0.0
    wy = -angmom[1] / ai[1]
    wz = -angmom[2] / ai[2]

    for i in range(len(w)):
        jx = i
        jy = i + 1
        jz = i + 2
        ang = np.array([
            wy * qq[jz] - wz * qq[jy],
            wz * qq[jx] - wx * qq[jz],
            wx * qq[jy] - wy * qq[jx]
        ])

        pang = ang * w[i]

        pp[jx] -= pang[0]
        pp[jy] -= pang[1]
        pp[jz] -= pang[2]

    qrot, prot = euler_rot(qq, pp)

    return (q, p)
