# Policy
#### The code is not public/published yet. If you use or want to use TSUNAMI, ask the developers and keep them involved ([Alessandro A. Trani](aatrani@gmail.com), [Mario Spera](mario.spera@live.it)).

# Install

#### Requirements:
For the C++ standalone code:  
- CMake (cmake >= 3.14)  
- (linux only) GNU C library (glibc >= 2.1)
- Compiler with C++17 support (e.g. GCC7 or later)

For the Python interface:  
- Python 3 develop libraries + numpy  
- SWIG (>=4.0)

### Instructions
CMake requires you to create a build directory; it can be the pre-existent `build` directory or any other user-created directory, anywhere else on the system.

To install, go into the build directory:   
```console 
$ cd build
```

Run the cmake command to initialize the current directory.  
The second argument must point to the directory where the file `CMakeLists.txt` is located (i.e. the top level of the source directory).  
If using the pre-existent `build` directory, `CMakeLists.txt` is in the parent directory:  
```console 
$ cmake .. <options>
```
The available compile options are described below.  

In the build folder, run make:  
```console 
$ make
```

The executable, parameter files and example input files will be located in the `bin/` subdirectory of the build directory.  

To run a quick test, go into the bin directory and run the TSUNAMI code.
```console
$ cd bin
$ ./tsunami.x
```

Something like this should appear:
```console
N = 3           t = 3.006627e+01     dt = 2.934776e+00      t/tfin= 100.22 %
```

You should see also the output files in the input folder:
```console
$ ls input
input_arc_energy.dat  input_arc_output.dat  [...]
```
The structure of the output and input files is described below.

# CMake options

Notice that true/yes/on/1 and false/no/off/0 are interchangable.

Beside the C++ executable, Tsunami comes a Python interface build with SWIG. The building of the Python interface must be enabled with CMake options

This repository also implements a solver for the secular equations of hierarchical triples, called Okinami, derived in Delaunay variables at the octupole approximation. It includes 1PN and 2PN precession, 2.5PN gravitational wave radiation, and tidal interactions for the inner orbit (see Blaes et al. 2002 and Naoz et al. 2013).  
Currently, there is no executable compiled to use Okinami, and the module is accessed only via the python interface. Thus, the option `-Dpython-true` is required.
To integrate the Kozai-Lidov equations we use a 8th order Dormand-Prince method with error estimation. 

#### - Python interface
```console 
-Dpython=true
```

Be sure to have installed SWIG and python3.  
To test the python interface, type from the build folder:
```console 
cd python
python3 test_tsunami.py
```
The test requires matplotlib.  
For more informations about the python interface, run and inspect the various scripts in the python folder.
- `test_tsunami.py` sets up a the Pythagorean 3-body problem, runs it, and shows the trajectories. Teaches the basics on how to create a particle set, run a simple Newtonian simulation, and plot the results.
- `test_tides.py` tests the tidal evolution for a binary, including spins, and compares it with the secular evolution.
- `test_okinami.py` tests Okinami, the secular evolution code for hierarchical triples, and compares it with Tsunami. Compare the resulting plot with Fig. 3 from Naoz et al. (2013).
- `test_mergers_collisions.py` is an example on how to deal with mergers and collisions, typically when post-Newtonians are enabled. It stops the simulation when two particles get closer than a multiple of their Schwarzschild radius, it substitutes them with their merger product, and then restarts the simulation.
- `triple_generator_nbody.py` sets up a hierarchical system from Keplerian orbital elements and evolves it with Tsunami
- `triple_generator_secular.py` same as `triple_generator_nbody.py`, but uses Okinami

#### - Compile the C++ executable 
```shell
-Dcpp=yes
```  
Default: cpp=yes (the esxecutable will be compiled)  

#### - Compile with spin support 
```shell
-Dspin=yes
```  
Default: spin=no (spins are not compiled)  

#### - Enable/disable timing
By default, TSUNAMI will print to screen an info line while running.
```console
N = 4           t = 2.827678e+05     dt = 1.097459e-03      t/tfin= 4.50 %
```
- $`N`$   is the number of particles
- $`t`$   is the current time
- $`dt`$  is the current physical timestep 
- $`t/tfin`$ is *percentage* of current time to final time   

The line is refreshed every output print out timestep (see Input / Output section). 

These informations can be useful when running manually a few simulations.  
However, whenever the output is redirected to a file (e.g. as a job on a HPC cluster), it can create lots of garbage lines.  
In this case, it is best to disable it with the following option:  

```console
-Dtiming=no
```
Default: timing=yes (On-the-fly run information will be printed out)

#### - Intel (or others) compiler

To change compiler, set the following flags in the cmake command: 

```console
-DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
```
TSUNAMI compiles correctly with Intel compilers (tested with icpc 19.0.0.117)


# Input / Output of the C++ executable

### Simulation parameters

TSUNAMI reads the simulation parameters from the file `input/tsunami_parameters.txt` of the working directory.  
Wherever you copy `tsunami.x`, remember to copy also the `input` folder to the same directory.  
 
Some of the input parameters can be overridden using command line options (see Command line section).

The parameters are described in the `tsunami_parameters.txt` file:

```txt
3          //number of particles
30      //total integration time (code units) //yr = 6.28372892847
1.0        //your mass unit (in solar masses)
1.0        //your length unit (in au)
1.0e-20     //time step for printing snapshots (x, y, z, vx, vy, vz, mass)
no        //do you want o include Post-Newtonian terms up to order 2.5 ? (yes or no)
no        //do you want to include equilibrium tides? (yes or no)
no        //do you want to include dynamical tides? (yes or no)
1.0       //X for checking for collisions when d < (r1+r2)*X. set to 0 to disable collisions
no       //do you want to include external potentials
```

##### Quick notes
- All these parameters can be overwritten using the command line arguments (see section below) without the need to edit the `tsunami_parameters.txt` file.
- Rembember to specify the correct number of particles.
- TSUNAMI uses N-body units assuming that G=1, M=MSun, L=au.   
  If you wish to provide different units for M and L, specify them in the parameter file or as command line option.   
  This means:  
     1) With PN terms off, the integration will be the same regardless of the provided units, 
        but the final integration time and output timesteps will be different. This is because the Newtonian N-body problem is time-scale-free, and our integration timestep is adaptive. 
     2) With PN terms on, we need the speed of light in the correct units, so check your units or the physics will be wrong
- Both total integration time and output printout are given in N-body units


### Command line
To display the command line help screen, run:  
```console 
$ ./tsunami.x -h
``` 

Command line arguments always overwrite the ones defined in `./input/tsunami_parameters.txt`.

Example: 
```console
$ ./tsunami.x input/solar_system.dat -N 9 -ft 6.28e2 -dt 6.28 -L au -fcol 1.5 -T -PN -c
Assuming 1 astronomical unit as length scale
Overriding parameter.txt: tides on
Overriding parameter.txt: PNs on
```
- **input/solar_system.dat**: read specified input file
- **-N 9**: Read 9 particles from input files (Pluto is not a planet)
- **-ft 6.28e2**: Integrate for 6.28e2 N-body time units
- **-dt 6.28**: write ouput every 6.28 N-body time units
- **-L au**: set 1 astronomical unit as unit length. Can be a number instead, in that case it is assumed to be in parsec units 
- **-fcol 1.5**: check for collisions when particles are distant less than 1.5 times the sum of the radii
- **-T**: enable equilibrium tides
- **-PN**: enable post-Newtonians
- **-c**: continue simulation from last output snapshot found in `solar_system_output.dat` (if any)

### Input files

By default, TSUNAMI reads the file `input/tsunami_default_input.dat` as input file.  
In this case, the output files are named `tsunami_default_input_energy.dat` and `tsunami_default_input_output.dat` in the same directory as the original input file.  

However, a different input file can be specified as *first positional argument* of `tsunami.x`:
```console
$ ./tsunami.x input/solar_system.dat -N 10 -ft 6.28e2 -dt 6.28 -L au
Assuming 1 astronomical unit as length scale
N = 10           t = 6.226881e+02     dt = 1.314902e-02      t/tfin= 99.15 %
```

In this case the output files are named `<filename>_output.dat` and `<filename>_energy.dat` 
and are created in the *input file directory*.
```console
$ ls input/
solar_system_energy.dat  solar_system.dat  solar_system_output.dat  [...]
```

Input files need to have the `.dat` extension.

Input file structure:
```txt

x  y  z   vx  vy  vz   mass  radius   particle_type   spin_x  spin_y  spin_z
                     <Particle 1>
                         ...
                     <Particle N>
```

- Radius is important for tides and particle collision, can be zero
- `particle_type` **must** be an integer (either positive or negative), 
  and it is used to assign tidel parameters and flags to each particle (see Tidal parameters section)

### Output files


The two output files are structured as follows.  
`<filename>_output.dat`:
```txt
x  y  z   vx  vy  vz   mass  spin_x spin_y spin_z
        <Particle 1 t0>
            ...
        <Particle N t0>
        <Particle 1 t1>
            ...
        <Particle N t1>
        ...............
        <Particle 1 tfin>
            ...
        <Particle N tfin>
```
- The ouput is always rescaled to the center of mass. The initial conditions will be also written as first ouput.
- Each line is a particle. Each block of N particles from the beginning corresponds to a time, given in `<filename>_energy.dat`
- The output file has Nparticles * Ntimesteps lines

`<filename>_energy.dat`:
```txt
time     dE/E0    energy  
        <t0>
        <t1>
        ...
        <tfin>
```
- Time and energy are in N-body units.
- $`dE/E0`$ is the energy error
- Each line corresponds to a time.
- The energy file has Ntimestep lines

# Tidal parameters

TSUNAMI implements the equilibrium tidal model from Hut 1981 and 
the dynamical tide drag-force from Samsing, Leigh & Trani 2018.   
By default, only the equilibrium tidal model is used.

File `input/tsunami_tide_table.dat` is a table that maps `particle_type` to tidal parameters.

Equilibrium tide:
- kaps: apsidal constant `kaps`, dimensionless
- taulag: time lag `taulag`, in seconds
- gyration radius: `rg`, dimensionless. Used to compute the inertia `I` = `M` (`R` `rg`)^2, where `M` and `R` are mass and radius of the particle  

Dynamical tide:
- polyIndex: polytropic index `gamma`

```txt
##############################################################################
### pType   taulag (in seconds)   kaps    polyIndex     gyration radius   
#############################################################################

# STARS
    0 	    0.15                0.014             3.0          0.25      # MS lowmass
    2 	    1e2     	        0.05              1.5	       0.2       # HGAP
    3 	    1.1e3               0.1               1.5	       0.35      # Giant branch
   14       0.0	                0.0               0.0	       0.0       # No tides (e.g. black hole)
# Default (hardcoded):
#           0.15                0.014             3.0          0.25 

# PLANETS
  100       4800e2              0.3               3.0	       0.1      # Rocky
  101       0.66e2              0.1               1.5	       0.15     # Giant 
# Default (hardcoded):
#           4800.0              0.3               3.0          0.1

# Testing values
 50 	   1e3       	        0.05              1.5          0.28
 110       5e7                  0.3               3.0          0.5
 120       5e8                  0.3               3.0          0.5
```

- `particle_type` must be an integer
- Numbers from 0 to 100 are "reserved" for stars
- Numbers from 100 to 1000 are "reserved" for planets
- Particle type between 0 and 1000 not found in the table are assigned 
  the default values reported in the comments
- For particle type outside 0-1000, tides are disabled 
  even if the corresponding particle type is included in the table.
  This allows to enable tides on a per-particle basis. 
  To disable tides you can also set kaps or taulag to zero or negative

The table is read at the beginning of the simulation.  
Each particle type found in the input file is then assigned the corresponding
tidal parameter.

In the Python interface this table is ignored, and the tidal parameters are simply assigned with the function initialize_tidal_parameters


