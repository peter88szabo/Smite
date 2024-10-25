from sampling.polyvibration import initialize_vibrational_modes
from sampling.polyvibration import specify_vib_modes 
from sampling.polyrotation  import initialize_rotational_modes
 
def init_vib_rot_modes(freq, init_vib_type='ZPE', init_rot_type='Jrot', **kwargs):
    temp = kwargs.get('temp', 300.0)
    jrot = kwargs.get('jrot', 0)
    krot = kwargs.get('krot', 0)

    #initialize all vibrational modes the same way (either with ZPE quantum state 'nvib = 0' or according to a temperature)
    vib_modes = initialize_vibrational_modes(freq=freq, init_vib_type=init_vib_type, temp=temp)

    rot_modes = initialize_rotational_modes(init_rot_type=init_rot_type, temp=temp, jrot=jrot, krot=krot)

    #then if any specific mode is fixed (energy, quantum number or temperature), everything in kwargs:
    if kwargs:
        vib_modes = specify_vib_modes(vib_modes, **kwargs)

    return (vib_modes, rot_modes)
