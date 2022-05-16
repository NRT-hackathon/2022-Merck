import numpy as np
import scipy.special as sp
import scipy.stats as stats
import dataFile_merck as dataFile
from random import *

###################################################
# Change numbers in this section to adjust system #
# parameters                                      #
###################################################

## define simulation unit as 1 um

drugD = 10
# from 1 um to 15 um
drugSTD = 0.05 ##### need to identify, just set as value for now
### may want to fit curve and pull out better?
drugAR = 10
# from 1 to 10 AR
drugLoad = 0.3
# from 30%w to 50%w
numStructures = 1
# number of structures to output for these conditions

# knowns from problem
implantD = 2000
implantL = 40000
###### implant is too large for us to model entire thing!!
######  change length to more reasonable length (doing disk)
implantD = 2000
implantL = 550
densityP = 0.929
densityD = 1.59

# account for rod dispersity in diameter
Ntypes = 11

#determine volume fraction of drug
vDrug = drugLoad*densityP/(densityD+drugLoad*densityP-drugLoad*densityD)
vPolymer = 1-vDrug
assert(np.abs(vDrug+vPolymer-1)<1e-5)
print(vDrug,vPolymer)

# Base Langevin thermostat damping parameter
tauDamp = 100.0
# timestep size (tau/dt)
deltat = 0.003
walloffset = 200

###################################################
# Calculations                                    #
###################################################

# ensure simulation box is larger than cylinder
lx = drugD*drugAR*4+implantD+walloffset
ly = drugD*drugAR*4+implantD+walloffset
lz = implantL

#approx number of rods
desiredV = np.pi*(implantD/2)**2*implantL*vDrug/drugAR
# doing entire cube now
desiredV = lx*ly*lz*vDrug/drugAR
nomrod = int(np.rint(desiredV/(4/3.*np.pi*(drugD/2.)**2)))

# polydisperse; create multiple rod diameters in space D with lognormal distribution
binmin = stats.lognorm.ppf(0.01,drugSTD,scale=drugD)
binmax = stats.lognorm.ppf(0.99,drugSTD,scale=drugD)
sizes = np.linspace(binmin,binmax,Ntypes)
bins = np.append(sizes-(sizes[1]-sizes[0])/2,sizes[-1]+(sizes[1]-sizes[0])/2)

# select rod diameters to fit distribution
print('Estimate of number of beads in simulation',nomrod,'\nIf over 1M, simulation may be too large')
#assert(nomrod<1e6)
r = stats.lognorm.rvs(size=nomrod,scale=drugD,s=drugSTD)
it = 1
while np.sum(((r<binmin) | (r>binmax))) > 0:
#    print('adjusting r, iteration',it)
    it += 1
    r = np.where(((r<binmin) | (r>binmax)),stats.lognorm.rvs(drugAR,size = 1, scale=drugD),r)
hist,edges = np.histogram(r,bins=bins)

tempVol = np.sum(hist*(4.0/3.0)*np.pi*(sizes/2.0)**3.0*drugAR)

# select number of rods to fit desired weight fraction of rods
err = abs(tempVol-desiredV)
errold = 0
while err > 0.0001*desiredV:
#    print('errors',err, desiredV/tempVol)
    hist = hist*desiredV/tempVol
    hist = hist.astype(int)
    tempVol = np.sum(hist*(4.0/3.0)*np.pi*(sizes/2.0)**3.0)
    errold = err
    err = abs(tempVol-desiredV)
    if err == errold:
        print('broken out')
        break
mask = hist == np.max(hist)
Nps  = np.array(hist)
#print('number rods',Nps)
#print('rod sizes',sizes)
N = np.sum(hist)
# normalize mass to the average diameter
masses = (sizes/2.0)**3/(drugD/2.)**3

# check rod volume fraction
check_vol = np.sum(4/3.*np.pi*(sizes/2.0)**3*Nps)
print("desired vol/achieved", desiredV/check_vol)

################### 
# confirm data file
###################
# place rods into cylinder to prevent overlap
#dataFile.dataFile(Nps, sizes, implantD/2., implantL, drugAR, masses)
dataFile.dataFile(Nps, sizes, lx/2.,lz, drugAR, masses)

### below here need to create lammps files to run system
# set simulation box and cylinder
toprint = []
toprint.append("region box block {} {} {} {} {} {} units box\n".format(-lx/2.0,lx/2.0,-ly/2.0,ly/2.0,-lz/2.0,lz/2.0))
#toprint.append("region cylinder cylinder z 0 0 {} EDGE EDGE units box side in open 1 open 2\n".format(implantD/2.+drugD*drugAR+walloffset,-implantL-drugAR*drugD,implantL+drugAR*drugD))
toprint.append("create_box "+str(Ntypes)+" box\n")
#toprint.append("region cylinder cylinder z 0 0 {} {} {} units box side in\n".format(lx/2.+walloffset,-lx/2.,lx/2.,implantD/2.+walloffset))
#toprint.append("create_box "+str(Ntypes)+" cylinder\n")
#toprint.append("region cylinder cylinder z 0 0 {} {} {} units box side in open 1 open 2\n".format(implantD/2.+drugD*drugAR+walloffset,-implantL/2,implantL/2))
#toprint.append("create_box "+str(Ntypes)+" cylinder\n")
f1 = open('lammps.boxcreate','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.close()

# set simulation method (Langevin dynamics)
scaleParam = (masses/sizes)
toprint = []
toprint.append("fix makeRidge1 all rigid/nve/small molecule\n")
langStr = "fix langevin all langevin 1.0 1.0 {} {} zero yes ".format(tauDamp,randint(10000,99999))
for typ in np.arange(1,Ntypes+1):
  langStr += "scale {} {} ".format(typ,scaleParam[typ-1])
langStr += "\n"
toprint.append("neigh_modify exclude molecule/intra all\n") 
toprint.append(langStr)
f1 = open('lammps.thermostat.sample','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.close()

# set simulation timestep size
toprint = []
toprint.append("timestep {}\n".format(deltat))
f1 = open('lammps.timestep','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.close()

# set drug crystal velocities randomly
toprint = []
toprint.append("velocity all create 1.0 {} dist gaussian units box\n".format(randint(10000,99999)))
f1 = open('lammps.velocity','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.write("run 0 \nvelocity all scale 1.0")
f1.close()

# set repulsive wall on cylinder to keep rods inside it
'''
toprint = []
for k,typ in enumerate(np.arange(1,Ntypes+1)):
  toprint.append("fix wall{}hi type{} wall/region cylinder harmonic 50.0 1.0 {}\n".format(typ,typ,sizes[k]/2.0+walloffset))
f1 = open('lammps.walldef','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.close()
'''

# set groups of particles 
toprint = [] 
for typ in np.arange(1,Ntypes+1):
  toprint.append("group type{} type {}\n".format(typ,typ))
f1 = open('lammps.group','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.close()

# set rod interactions to purely repulsive
toprint = []
for k,typ1 in enumerate(np.arange(1,Ntypes+1)):
  for j,typ2 in enumerate(np.arange(typ1,Ntypes+1)):
    string = "pair_coeff {} {} ".format(typ1,typ2)
    string += "1.0 {:.2f} ".format((sizes[typ1-1]+sizes[typ2-1])/2.0+0)
    cutoff = 2.0**(1.0/6.0)*((sizes[typ1-1]+sizes[typ2-1])/2.0+0)
    string += "{:.2f}".format(cutoff)
    toprint.append(string+"\n")
f1 = open('lammps.pair_coeff','w')
for pline in toprint:
  f1.write("%s" % pline)
f1.close()

# create initial lammps script to read in rods and do inital equilibration
toprint = []
toprint.append('timer timeout 47:40:00 every 100000\nunits       lj\n')
toprint.append('atom_style 	bond\nboundary    p p p\npair_style	lj/cut/opt {}\n'.format(drugD*drugAR))
toprint.append('pair_modify mix arithmetic shift yes\ninclude lammps.boxcreate\n')
toprint.append('include lammps.group\n')
toprint.append('include lammps.pair_coeff\ninclude lammps.timestep\nrun_style     	verlet\ninclude lammps.group\nread_data dataFile add append\n')
#toprint.append('include lammps.walldef\ninclude lammps.group\nread_data dataFile add append\n')
toprint.append('thermo 	10000\nthermo_style 	custom step temp epair pe etotal lx ly lz vol pxx pyy pzz press\nthermo_modify 	norm no\nfix 		thermo_print all print 10000 "$(step) $(temp) $(epair) $(pe) $(etotal) $(lx) $(ly) $(lz) $(vol) $(pxx) $(pyy) $(pzz) $(press) " append thermo.out screen no title "# step temp epair pe etotal lx ly lz vol pxx pyy pzz press "\n')
toprint.append('include 	lammps.velocity\ninclude 	lammps.thermostat.sample\n')
toprint.append('neigh_modify 	exclude molecule/intra all\ncomm_modify 	cutoff {}\n'.format(np.max(sizes)*drugAR))
#toprint.append('dump 1 all custom 1 output.txt id mol type x y z\ndump_modify 1 sort id append yes\n')
#toprint.append('run 0\n')
#toprint.append('dump 1 all custom 200000 output.txt id mol type x y z\ndump_modify 1 sort id append yes\n')
toprint.append('run {}\nwrite_data 	data.inter\nwrite_restart 	restart.inter\n'.format(int(np.rint(200000))))
toprint.append('print $(step) file init_done.txt\n')
toprint.append('dump 1 all custom 200000 output.txt id mol type x y z\ndump_modify 1 sort id append yes\n')
toprint.append('run {}\nwrite_data 	data.inter\nwrite_restart 	restart.inter\n'.format(int(np.rint(1000000))))
toprint.append('print $(step) file simdone.txt\n')
f1 = open('lammps.run','w')
for pline in toprint:
    f1.write("%s" % pline)
f1.close()

# dump diameters of particles to help with analysis
toprint = []
for i in range(1,Ntypes+1):
    string = "{} {} {} ".format(i, sizes[i-1], sizes[i-1]/2)
    toprint.append(string+"\n")
f1 = open('particle.diameters','w')
for pline in toprint:
    f1.write("%s" % pline)
f1.close()
