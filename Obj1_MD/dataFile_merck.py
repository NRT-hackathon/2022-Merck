import math
import random
import numpy as np
import scipy.spatial as spatial
def dataFile(nrods, sizes, radius,length, AR, masses, fileName = "dataFile"):
    #give total num of atom types
    types = len(sizes)
    fileO = open(fileName, 'w')
    # gives total number of particles for dispersity
    numAtoms = int(np.sum(AR*nrods))
    toPrint = "#C Heil created LAMMPS dataFile\n" 
    fileO.write(toPrint) 

    toPrint = str(numAtoms) + " atoms\n" #prints the number of atoms at the top of the dataFile.
    fileO.write(toPrint)

    #For the rigid 1-comp nanorod simulations all of these are non-existant so it is all 0
    toPrint = "0 bonds \n"+str(len(sizes))+" atom types \n0 bond types \n0 angles \n0 angle types \n0 dihedrals \n0 dihedral types \n0 impropers \n0 improper types"
    
    fileO.write(toPrint)
    
    fileO.write("\n\nMasses\n\n")  #Mass section

    # trying to allow for more than 1 mass
    for k in range(types):
        fileO.write(str(k+1)+ " " + str(masses[k]) +" \n") 
    
    fileO.write("\n\n")

    #Atoms Section using bond style
    #Atom-ID molecule-ID atom-Type x y z

    fileO.write("Atoms\n\n")
    #Takes in the dimensions that I need to create the inscribed cube in the supraball and generate the cells for the rod creation.
    # edit to make it deal with largest radius
    r_max = np.max(sizes/2.)*1.005
    boxlength = radius/1.
    xmin = -boxlength + r_max 
    xmax = boxlength - r_max 
    ymin = -boxlength + r_max
    ymax = boxlength - r_max
    zmin = -length/2 + np.max(sizes)*AR/2
    zmax =  length/2 - np.max(sizes)*AR/2

    r_max = np.max(sizes)*1.005
    # finds max number of cells in each direction
    numCellsx = int((xmax-xmin)/(r_max))
    numCellsy = int((ymax-ymin)/(r_max))
    #Range of generation cube in the x dimension
    x_range = np.linspace(xmin, xmax, numCellsx) 
    #Range of generation cube in the y dimension
    y_range = np.linspace(ymin, ymax, numCellsy) 
    #repeated slabs in the z dimension
    # finds max number of cells in y direction
    numCellsz = int((zmax-zmin)/(np.max(sizes)*AR*1.01))
    z_range = np.linspace(zmin, zmax, numCellsz) 

    genMatrix = np.zeros((numCellsx*numCellsy*numCellsz))
#    print("total cells", len(genMatrix),"required", np.sum(nrods))
    if len(genMatrix) < np.sum(nrods):
        print('number of cells',len(genMatrix),'number needed',np.sum(nrods))
        print("need more cells")
        more_cells_needed

    count = 0
    for i in range(types):
        genMatrix[count:count+nrods[i]] = i+1
        count += nrods[i]

    #This shuffles the genMatrix so all the ones aren't right next to eachother 
    np.random.shuffle(genMatrix)

    #Creates an array of every possible coordinate in the inscribed cube (based off of the cells that I created. 
    possibleCoordinates = cartesian((x_range, y_range, z_range))

    # initial values for atom num, mol num, and rods placed
    moleculeCounter = 1
    atomCounter = 1
    bodyCounterMe = 0

    #pos = np.zeros((len(genMatrix),3))
    #Creates atoms and their locations. 
    #need to organize atom type 1-N
    for j in range(types):
        #genMatrix locations that have the correct atom ID
        locs = np.where(genMatrix==(j+1))
        # transpose to be able to operate on it
        locs = np.transpose(locs)
        # rod length
        rodL = AR
        for index, bodies in enumerate(locs):
            # spot gives genMatrix location
            spot = bodies[0]
            # counts
            bodyCounterMe = bodyCounterMe + 1
            # retrieves coordinates of the starting bead
            x1 = (possibleCoordinates[spot][0]) 
            y1 = (possibleCoordinates[spot][1])
            z1 = (possibleCoordinates[spot][2])  
            # for the length of the rod, place beads
            for i in range(rodL):
                #pos[atomCounter,:] = [x1,y1,z1+sizes[k]*i]
                toPrint = str(atomCounter) + " " + str(moleculeCounter) + " " 
                toPrint += str(j+1)+" " + str(x1) + " "
                toPrint += str(y1) + " " + str(z1+sizes[k]*i) + "\n"
                # write to file and update atom num & flag
                fileO.write(toPrint)
                atomCounter = atomCounter + 1
            moleculeCounter = moleculeCounter + 1           
    
    #tree = spatial.KDTree(pos)
    #print('overlaps',np.sum(tree.query_pairs(np.min(sizes)*0.95)))  
    print('rods placed',bodyCounterMe)

#credit for cartesian function to user pv on stackoverflow.com
#https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    ### need to check this math

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


