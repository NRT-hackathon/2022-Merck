# Merck
CHEG867 Class Project

All code written and developed by C. Heil. 

For Objective 1: create an implant structure with specific drug crystal design parameters (drug loading, diameter, aspect ratio, and dispersity).

To create drug implant structures with defined implant size and drug characteristics (loading, diameter, aspect ratio, dispersity in diameter), utilize the Obj1_MD folder. The merck.implant.model.create.py file enables the creation of all files required to run an MD simulation with the specified parameters set.

The dataFile_merck.py file places the drug crystals/rods into the simulation box. If one runs into an issue with this method saying that there are not enough rod/drug placements available, first try adjusting the implant length or width/height to create more placements. If the issue continues, one may need to more intelligently place the drug by changing how this file discretizes the space for rod/drug placement. This is certainly a direction for future work as highly disperse drug crystals will require adjustments to the rod/drug placement.

For Objective 2: simulate drug release from the implant structure created in Objective 1.

Once a LAMMPS MD structure has been generated for the drug design parameters in Objective 1, the structure must be converted into a grid-based representation to utilize the finite difference method to solve for drug release. The code_preprocess_lammps_fd.py file performs this step. It enables a user to specify the desired cylindrical implant radius as well to set if the user would prefer an annulus to ignore inner points (useful if the user aims to fit the early-time drug release profile to predict long-time drug release). This step only needs to be performed once to obtain the grid-based representation of the structure for input into the FD algorithm. Future directions for improvements include better edge detection for the drug crystals to ensure grid points near the drug crystal are given the appropriate drug mass considering how much crystal is present around the grid point.

The FD algorithm is written in code_fd_3d.py to solve the drug dissolution (Noyes-Whitney) and diffusion (Fick's) equations over time. The code was developed to enable restarting enabling runs to continue when stopped instead of restarting. The FD algorithm was written in python and allows for the use of multiple CPUs to speed-up the method. Future directions for improvement include changing the programming language to C, Fortran, etc. to enable faster calculations and enabling multiprocessing on CPUs/GPUs. The output of interest from the file 'fd-conc-release.txt' which provides the instantaneous drug release over a set time interval. 


Code provided as is with no promise of functionality.
