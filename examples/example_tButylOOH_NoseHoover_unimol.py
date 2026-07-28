from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == '__main__':
    import random
    from smite import Fragment

    # xtb optimized (with spinpol) terc-Butyl-OOH
    xyz = '''
    C           -0.00498503532174        0.00867693128428       -0.01937965630342
    C           -0.09777584431124        0.16953427015808        1.45743592478362
    C            1.45446349024161        0.00553411604699       -0.48516658053248
    C           -0.73203950683230       -1.25680101858370       -0.48513176697136
    O           -0.68022342909887        1.17821684396525       -0.50473196837116
    H            1.48952145911009       -0.04551391515466       -1.57024548299753
    H            1.95007440141733        0.91752942543206       -0.16153673636857
    H            1.97343745463074       -0.85251287550408       -0.06792945438035
    H           -0.70539861034874       -1.31268360637506       -1.57021108562863
    H           -0.24847245145021       -2.13529274083479       -0.06790624694383
    H           -1.76965032515023       -1.22996201200190       -0.16148079420123
    H           -0.96626289399235       -0.17789070525963        1.98249448443088
    H            0.63666904906233        0.74834508464096        1.98227940504432
    O           -0.69675420854804        1.20688571491409       -1.94150986067637
    H           -1.17220254940838        2.03047248727211       -2.10262118088386
     '''

    qcinput_XTB = {
    'qchem': 'XTB',
    'path': '/home/peter/Programs/orca_6_1_1_linux_x86-64_shared_openmpi418/xtb',
    'nproc': 4,
    'functional': '',
    'basis': '',
    'charge': 0,
    'multiplicity': 2,
    'additional': '--iterations 200 --tblite',
    'wfu': False
    }

    fix_quantum = [(38, 2)]

    seed = 13349112
    random.seed(seed)

    QOOH = Fragment.Polyatom_Init(fname='TBOOH_NoseHoover', qchem=qcinput_XTB, xyz=xyz, random_rot=False)
    QOOH.Specify_Mode_Sampling(init_vib_type='ZPE', init_rot_type='Jfix', jrot=0, fix_quantum=fix_quantum)

    print('\nNose-Hoover NVT equilibration at 500 K:')
    QOOH.sample_and_run_trajectory(
        traj_file='TBOOH_NoseHoover_NVT_equilibration.xyz',
        integrator='leapfrog',
        integrator_order=4,
        timestep=1.0,
        maxstep=500,
        iprint=1,
        Rstop=10.0,
        thermostat='nosehoover',
        thermo_param=20.0,
        thermo_temp=500.0,
    )

    print('\nNose-Hoover NVT production at 500 K:')
    QOOH.run_trajectory(
        traj_file='TBOOH_NoseHoover_NVT_production.xyz',
        integrator='leapfrog',
        integrator_order=4,
        timestep=1.0,
        maxstep=1000,
        iprint=1,
        Rstop=10.0,
        thermostat='nosehoover',
        thermo_param=20.0,
        thermo_temp=500.0,
    )

    print('\nNVE segment after Nose-Hoover thermalization:')
    QOOH.run_trajectory(
        traj_file='TBOOH_NoseHoover_NVE_final.xyz',
        integrator='leapfrog',
        integrator_order=4,
        timestep=1.0,
        maxstep=3000,
        iprint=1,
        Rstop=10.0,
        spectrum=True,
    )

    QOOH.vibrational_spectrum(dt=1.0, print_maxfreq=7000.0)
