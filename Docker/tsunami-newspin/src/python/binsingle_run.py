"""
binsingle_run.py

This script runs a single row of initial conditions from a hdf5 file previously generated with binsingle_initial_conditions.py
The core of the integration and analysis of the binary-single evolution is done in the script binsingle_evolve_system.py
This script orchestrates the initialization of the simulations and the storing of the outcome data.

This script is usually run automatically via the bash script <setname>.sh, which will create a list of command lines
<setname>_cmdrun.txt, where each line is contains a call to this script.

The outcome data will be saved as an individual raw in the file <setname>_output.txt, unless the -I option is turned on.
With -I, the outcome data is saved into an individual file, <setname>_<rownumber>_out.txt, where <rownumber> is the number
corresponding to the simulation row in the initial condition file.
The option -I is mainly used for debugging individual runs.

If there is a merger or collision during a simulation, the pre-collision coordinates will be saved into the folder <setname>_coll/

If this script is run via the bash script <setname>.sh, the txt output file is eventually converted into hdf5 file
by the script binsingle_txt_to_h5.py

Some initial condition parameters can be set in this script, namely:
    * The final integration time (-tfin)
    * The initial distance between the binary CoM and the single (-idist)
    * The collision radius of individual particles (-Rschw)

Check the varius input parameters in the argument options with -h

The outcome properies are described in binsingle_evolve_system.py
"""

import tsunami
import numpy as np
import pandas as pd
import argparse
from binsingle_evolve_system import setup_system, run_system
from binsingle_initial_conditions import output_suff, coll_folder_suff

KU = tsunami.KeplerUtils()
kmap = {"begin": 0, "escape": 1, "breakup": 2, "interacting": 3, "excursion": 4, "will_interact": 5}


def read_write(filein, fileout, ind, save_traj=False, logging=False, individual=False,
               Rschw_mult=10, idist_mult=150, tfin_mult=1e3, timing=False):
    icdata = pd.read_hdf(filein)
    ici = icdata.loc[ind]
    fname = "{:07d}".format(ind)

    print("\nid =", ind)
    print("m1, m2, m3 = {:3.2f}, {:3.2f}, {:3.2f}".format(ici.m1, ici.m2, ici.m3))
    print("a_inn, e_inn = {:3.2f}, {:3.2f}".format(ici.a_inn, ici.e_inn))
    print("ome_inn, nu_inn = {:3.2f}, {:3.2f}".format(np.degrees(ici.ome_inn), np.degrees(ici.nu_inn)))

    print("b, b/a_inn = {:3.2f}, {:3.2f}".format(ici.b, ici.b / ici.a_inn))

    p, v, m, R, orb_inn, orb_out, time_to_enc = setup_system(m1=ici.m1, m2=ici.m2, m3=ici.m3,
                                                             a_inn=ici.a_inn, e_inn=ici.e_inn,
                                                             ome_inn=ici.ome_inn, nu_inn=ici.nu_inn,
                                                             vinf=ici.vinf, b=ici.b, i_out=ici.i_out,
                                                             ome_out=ici.ome_out, Ome_out=ici.Ome_out,
                                                             Rschw_mult=Rschw_mult, d_sin_a_inn=idist_mult)

    # input()

    t_stop = time_to_enc * tfin_mult
    status = run_system(p=p, v=v, m=m, R=R,
                        ftime=t_stop, dtout=0.00001,
                        time_to_enc=time_to_enc,
                        save_traj=save_traj,
                        logging=logging,
                        fname=fname,
                        timing=timing,
                        setname=filein.replace(".h5", ""))

    print(status)

    # Input variables as output
    m1_i, m2_i, m3_i = m
    orb_inn_i = orb_inn
    orb_out_i = orb_out

    # Output variables
    k = kmap[status.status]
    Nexcurs = status.Nexcurs
    Nexchan = status.Nexchan
    coll = status.collision
    orb_inn_f = status.orb_inn
    orb_out_f = status.orb_out
    begin_time = status.begin_time
    end_time = status.end_time
    ibin, jbin, ksin =status.ibin[0], status.ibin[1], status.isin
    i_c, j_c = status.collid[0], status.collid[1]

    if individual: fileout = fileout.replace(output_suff, "_" + fname + "_out.txt")

    with open(fileout, 'a') as f:
        f.write("{:d}   ".format(ind))

        f.write("{:d} {:d} {:d} {:d}  ".format(k, Nexcurs, Nexchan, coll))
        f.write("{:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e}   ".format(*orb_inn_f))
        f.write("{:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e}   ".format(*orb_out_f))
        f.write("{:1.16e} {:1.16e} {:1.16e}   ".format(begin_time, end_time, time_to_enc))
        f.write("{:d} {:d} {:d}   ".format(ibin, jbin, ksin))

        f.write("{:d} {:d}   ".format(i_c, j_c))
        f.write("{:1.16e} {:1.16e} {:1.16e}   ".format(m1_i, m2_i, m3_i))
        f.write("{:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e}   ".format(*orb_inn_i))
        f.write("{:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e} {:1.16e}   ".format(*orb_out_i))

        f.write("\n")


def option_parser():
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("setname", help="setname or h5 filename",
                        type=str)
    result.add_argument("-id", help="simulation number",
                        dest="id", type=int)
    result.add_argument("-S", help="save trajectories",
                        dest="save_traj", action="store_true")
    result.add_argument("-L", help="save logfile",
                        dest="logging", action="store_true")
    result.add_argument("-I", help="individual save",
                        dest="individual", action="store_true")
    result.add_argument("-T", help="print timing",
                        dest="timing", action="store_true")

    result.add_argument("-tfin", help="final time, in units of time_to_encounter",
                        dest="tfin_mult", type=float, default=1e3)
    result.add_argument("-idist", help="initial distance, in units of binary semimajor axis",
                        dest="idist_mult", type=float, default=5)
    result.add_argument("-Rschw", help="collision radius, in units of Schwarzschild radii",
                        dest="Rschw_mult", type=float, default=10)
    return result


if __name__ == "__main__":
    args = option_parser().parse_args()
    input_file = args.setname
    if not input_file.endswith(".h5"): input_file = input_file + ".h5"
    output_file = input_file.replace(".h5", output_suff)
    coll_folder = input_file.replace(".h5", coll_folder_suff)

    read_write(filein=input_file, fileout=output_file, ind=args.id,
               save_traj=args.save_traj, logging=args.logging, individual=args.individual,
               Rschw_mult=args.Rschw_mult, idist_mult=args.idist_mult, tfin_mult=args.tfin_mult,
               timing=args.timing)
