import math
import numpy as np

def get_angle(vec_A, vec_B, tol=1e-5):
    vec_A_mag = np.linalg.norm(vec_A)
    vec_B_mag = np.linalg.norm(vec_B)

    if (vec_A and vec_B) is None or vec_A_mag <= tol or vec_B_mag <= tol:
        return None

    arg = np.dot(vec_A, vec_B) / (vec_A_mag * vec_B_mag)
    arg = min(1.0, max(-1.0, arg))

    return math.degrees(math.acos(arg))


def scattering_angles(vrel_ini, vrel_fin, Lorb_ini, Lorb_fin, Jrot_ini_A, Jrot_fin_A, Jrot_ini_B, Jrot_fin_B):
    '''
    if a vector quantity is not defined for an atom
    (as the rotational angmom (Jrot_ini_A or Jrot_ini_B)
    then those qunatties have a None value

    fragment: is the number of fragments



C     CALCULATE RELATIVE A+B PROPERTIES
C
      RDMASS=WTA(NPATH)*WTB(NPATH)/(WTA(NPATH)+WTB(NPATH))
      RCM=0.0D0
      DO 11 I=1,3
      QR(I)=QCMA(I)-QCMB(I)
      VR(I)=VCMA(I)-VCMB(I)
      RCM=RCM+QR(I)**2
   11 CONTINUE
      RCM= SQRT(RCM)
      VREL=0.0D0
      VRELSQ=0.0D0
      DO 12 I=1,3
      VREL=VREL+VR(I)*QR(I)
      VRELSQ=VRELSQ+VR(I)**2
   12 CONTINUE
      VREL=VREL/RCM
      EREL=RDMASS*VREL**2/2.0D0/C1
      ERELSQ=RDMASS*VRELSQ/2.0D0/C1
      OAM(1)=(QR(2)*VR(3)-QR(3)*VR(2))*RDMASS
      OAM(2)=(QR(3)*VR(1)-QR(1)*VR(3))*RDMASS
      OAM(3)=(QR(1)*VR(2)-QR(2)*VR(1))*RDMASS
      OAM(4)= SQRT(OAM(1)**2+OAM(2)**2+OAM(3)**2)
      BF=OAM(4)/RDMASS/ SQRT(VRELSQ)

      IF(NATOMB(1).EQ.0)GO TO 25
      VRELSQ= SQRT(VRELSQ)
      DUM=(VI(1)*VR(1)+VI(2)*VR(2)+VI(3)*VR(3))/VI(4)/VRELSQ
      IF(DUM.GT.1.00D0) DUM=1.00D0
      IF(DUM.LT.-1.00D0) DUM=-1.00D0
      VRELSQ=VRELSQ**2


    '''

    # Initialize scattering angles
    ang = [None] * 20  

    """
    2-vector correlations from:
    vrel_ini  vrel_fin
    Lorb_ini  Lorb_fin
    """
    #vrel_ini, vrel_fin
    ang[0] = get_angle(vrel_ini, vrel_fin, tol=1e-6)

    #Lorb_ini, Lorb_fin
    ang[1] = get_angle(Lorb_ini, Lorb_fin, tol=1e-5)

    #vrel_ini, Lorb_ini
    ang[2] = get_angle(vrel_ini, Lorb_ini, tol=1e-6)

    #vrel_ini, Lorb_fin
    ang[3] = get_angle(vrel_ini, Lorb_fin, tol=1e-6)

    #vrel_fin, Lorb_ini
    ang[4] = get_angle(vrel_fin, Lorb_ini, tol=1e-5)

    #vrel_fin, Lorb_fin
    ang[5] = get_angle(vrel_fin, Lorb_fin, tol=1e-5)



    """
    2-vector correlations from:
    vrel_ini  vrel_fin
    Jrot_ini_A  Jrot_fin_A
    """
    #Jrot_ini_A, Jrot_fin_A,
    ang[6] = get_angle(Jrot_ini_A, Jrot_fin_A, tol=1e-5)

    #vrel_ini, Jrot_ini_A
    ang[7] = get_angle(vrel_ini, Jrot_ini_A, tol=1e-5)

    #vrel_ini, Jrot_fin_A
    ang[8] = get_angle(vrel_ini, Jrot_fin_A, tol=1e-6)

    #vrel_fin, Jrot_ini_A
    ang[9] = get_angle(vrel_fin, Jrot_ini_A, tol=1e-5)

    #vrel_fin, Jrot_fin_A
    ang[10] = get_angle(vrel_fin, Jrot_fin_A, tol=1e-5)



    """
    2-vector correlations from:
    vrel_ini  vrel_fin
    Jrot_ini_B  Jrot_fin_B
    """
    #Jrot_ini_B, Jrot_fin_B,
    ang[11] = get_angle(Jrot_ini_B, Jrot_fin_B, tol=1e-5)

    #vrel_ini, Jrot_ini_B
    ang[12] = get_angle(vrel_ini, Jrot_ini_B, tol=1e-5)

    #vrel_ini, Jrot_fin_B
    ang[13] = get_angle(vrel_ini, Jrot_fin_B, tol=1e-6)

    #vrel_fin, Jrot_ini_B
    ang[14] = get_angle(vrel_fin, Jrot_ini_B, tol=1e-5)

    #vrel_fin, Jrot_fin_B
    ang[15] = get_angle(vrel_fin, Jrot_fin_B, tol=1e-5)


    """
    2-vector correlations from:
    Jrot_ini_A  Jrot_fin_A
    Jrot_ini_B  Jrot_fin_B
    """
    #Jrot_ini_A, Jrot_ini_B,
    ang[16] = get_angle(Jrot_ini_A, Jrot_ini_B, tol=1e-5)

    #Jrot_fin_A, Jrot_fin_B,
    ang[17] = get_angle(Jrot_fin_A, Jrot_fin_B, tol=1e-5)

    #Jrot_ini_A, Jrot_fin_B,
    ang[18] = get_angle(Jrot_ini_A, Jrot_fin_B, tol=1e-5)

    #Jrot_fin_A, Jrot_ini_B,
    ang[19] = get_angle(Jrot_fin_A, Jrot_ini_B, tol=1e-5)

    """
    2-vector correlations from:
    Jrot_ini_A  Jrot_fin_A
    Lorb_ini  Lorb_fin
    """
    #Jrot_ini_A, Lorb_ini,
    ang[20] = get_angle(Jrot_ini_A, Lorb_ini, tol=1e-5)

    #Jrot_fin_A, Lorb_ini,
    ang[21] = get_angle(Jrot_fin_A, Lorb_ini, tol=1e-5)

    #Jrot_ini_A, Lorb_fin,
    ang[22] = get_angle(Jrot_ini_A, Lorb_fin, tol=1e-5)

    #Jrot_fin_A, Lorb_fin,
    ang[21] = get_angle(Jrot_fin_A, Lorb_fin, tol=1e-5)


    """
    2-vector correlations from:
    Jrot_ini_B  Jrot_fin_B
    Lorb_ini  Lorb_fin
    """
    #Jrot_ini_B, Lorb_ini,
    ang[20] = get_angle(Jrot_ini_B, Lorb_ini, tol=1e-5)

    #Jrot_fin_B, Lorb_ini,
    ang[21] = get_angle(Jrot_fin_B, Lorb_ini, tol=1e-5)

    #Jrot_ini_B, Lorb_fin,
    ang[22] = get_angle(Jrot_ini_B, Lorb_fin, tol=1e-5)

    #Jrot_fin_B, Lorb_fin,
    ang[23] = get_angle(Jrot_fin_B, Lorb_fin, tol=1e-5)


    return ang


def calculate_dihedral_angle(vini, vfin, jrot):
    '''
    Calculate the dihedral angle between the (k x k') and (k x j) planes.
    '''
    jrot_mag = np.linalg.norm(jrot)
    vini_mag = np.linalg.norm(vini)
    vfin_mag = np.linalg.norm(vfin)

    if vini_mag <= 1e-6 or vfin_mag <= 1e-5 or jrot_mag <= 1e-5:
        return None


    vn = normal_vector_of_collision_plane(vini, vfin)

    # Calculate the dot product of VR and AMBI, then normalize by their magnitudes
    arg = (vr[0] * ambi[0] + vr[1] * ambi[1]) \
        / (np.sqrt(vr[0]**2 + vr[1]**2) * np.sqrt(ambi[0]**2 + ambi[1]**2))

    arg = min(1.0, max(-1.0, arg))

    cosgamma = np.dot(vn, jrot) / jrot_mag #vn is already normalized

    '''
      this dot product (actually the sign of it)
      <k x k'|j'> = cos(gamma)
      will help us to decide wich side of the (k x k') plane we are
    '''
    if cosgamma >= 0.0:
        ang = np.degrees(np.arccos(arg))
    else:
        ang = 360.0 - np.degrees(np.arccos(arg))

    return ang




def normal_vector_of_collision_plane(vini, vfin):
    '''
    Find normal unit vector to collision plane (vrel_ini x vrel_final)
    '''

    vn = np.cross(vini, vfin)

    norm = np.linalg.norm(vn)

    if norm != 0.0:
        vn /= norm

    return vn


