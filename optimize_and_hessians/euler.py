import numpy as np;
import random
from math import sin, cos, pi

def euler_rot(qq, pp):
    phi   = random.uniform(0, 2 * pi)
    theta = math.acos(random.uniform(0, 1))
    chi   = random.uniform(0, 2 * pi)

    rxx =  cos(theta)*cos(phi)*cos(chi) - sin(phi)*sin(chi)
    rxy = -cos(theta)*cos(phi)*sin(chi) - sin(phi)*cos(chi)
    rxz =  sin(theta)*cos(phi)
    ryx =  cos(theta)*sin(phi)*cos(chi) + cos(phi)*sin(chi)
    ryy = -cos(theta)*sin(phi)*sin(chi) + cos(phi)*cos(chi)
    ryz =  sin(theta)*sin(phi)
    rzx = -sin(theta)*cos(chi)
    rzy =  sin(theta)*sin(chi)
    rzz =  cos(theta)

    q = [0] * len(qq)
    p = [0] * len(pp)

    for i in range(len(qq)/3):
        q[3*i    ] = qq[3*i]*rxx + qq[3*i+1]*rxy + qq[3*i+2]*rxz
        q[3*i + 1] = qq[3*i]*ryx + qq[3*i+1]*ryy + qq[3*i+2]*ryz
        q[3*i + 2] = qq[3*i]*rzx + qq[3*i+1]*rzy + qq[3*i+2]*rzz

        p[3*i    ] = pp[3*i]*rxx + pp[3*i+1]*rxy + pp[3*i+2]*rxz
        p[3*i + 1] = pp[3*i]*ryx + pp[3*i+1]*ryy + pp[3*i+2]*ryz
        p[3*i + 2] = pp[3*i]*rzx + pp[3*i+1]*rzy + pp[3*i+2]*rzz
    return (q,p)

