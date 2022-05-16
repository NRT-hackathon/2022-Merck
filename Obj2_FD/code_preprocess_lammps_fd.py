## LAMMPS file processing code written by C. Heil; UDel Spring 2022
import sys
import numpy as np
import scipy.spatial as spatial
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def select_grid(pos, diameters, connectors, connectors_surf, dia_con, dia_con_surf, gridpos, dcutoff, dx):
#def select_grid(pos, diameters, connectors, gridpos, dcutoff, dx):
    mask = np.zeros((len(gridpos)),dtype='int')
    tree = spatial.KDTree(gridpos)

    ## first go through connectors and assign grid points to them
    ## set their mask value to 1 to indicate connector spots
    ## doing this first enables the next part to overwrite as needed
    # go through connects and get grid points within
    for i, element in enumerate(connectors):
        #print('con',element)
        #print('dia',dia_con[i])
        # dcutoff/2. away (cubic though)
        # get each point's neighbors
        distance = np.sqrt(3)*np.max([dia_con[i]/4.,dcutoff/2.])
        neigh = np.array(tree.query_ball_point(element,distance+dx/2))
        # only consider neigh that are cubic away
        if len(neigh)==0:
            continue
        pos_temp = gridpos[neigh]
        if len(pos_temp)==0:
            continue
        delta = np.max(np.abs(element-pos_temp),axis=1)
        #neigh = neigh[delta<dcutoff/2.+dx/2.]
        neigh = neigh[delta<np.max([dia_con[i]/4.,dcutoff/2.])+dx/2.]
        if len(neigh)==0:
            continue
        mask[neigh] = 1
    
    for i, element in enumerate(connectors_surf):
        # dcutoff/2. away (cubic though)
        # get each point's neighbors
        distance = np.sqrt(3)*dia_con_surf[i]/2.
        neigh = np.array(tree.query_ball_point(element,distance+dx/2))
        # only consider neigh that are cubic away
        if len(neigh)<1:
            continue
        pos_temp = gridpos[neigh]
        if len(pos_temp)==0:
            continue
        delta = np.max(np.abs(element-pos_temp),axis=1)
        neigh = neigh[delta<dia_con_surf[i]/2.+dx/2.]
        if len(neigh)==0:
            continue
        mask[neigh] = 1

    ## then go through pos and assign grid points to them
    ## set their mask value to 2 to indicate drug crystal
    for i, element in enumerate(pos):
        distance = np.sqrt(3)*(diameters[i]/2.)
        # dcutoff/2. away (cubic though)
        # get each point's neighbors
        neigh = np.array(tree.query_ball_point(element,distance+dx/2.))
        # only consider neigh that are cubic away
        if len(neigh)<1:
            continue
        pos_temp = gridpos[neigh]
        delta = np.max(np.abs(element-pos_temp),axis=1)
        neigh = neigh[delta<diameters[i]/2.+dx/2.]
        if len(neigh)==0:
            continue
        mask[neigh] = 2
    
    ## now link grid points together using dictionary named graph
    gridpos = gridpos[mask>0]
    tree = spatial.KDTree(gridpos)
    graph = {}
    for i, ind in enumerate(gridpos):
        # get each point's neighbors
        neigh = tree.query_ball_point(ind,dx*1.01)
        
        # ignore the point itself
        selfp = -1
        ## address issue of not having a neighbor(s) in certain dims 
        x = []
        y = []
        z = []
        for j, element in enumerate(neigh):
            if element == i:
                selfp = j
                continue
            deltar = np.abs(ind-gridpos[element])
            if deltar[0] > 0:
                x.append(element)
            elif deltar[1] > 0:
                y.append(element)
            elif deltar[2] > 0:
                z.append(element)
        del neigh[selfp]

        # if only have 1 neigh, then need to double it
        # if have no neigh, then need to add self (to keep conc_update correct)
        if len(x) <2:
            if len(x) == 1:
                neigh.append(x[0]) 
            else:
                neigh.append(i) 
                neigh.append(i) 
        if len(y) <2:
            if len(y) == 1:
                neigh.append(y[0]) 
            else:
                neigh.append(i) 
                neigh.append(i) 
        if len(z) <2:
            if len(z) == 1:
                neigh.append(z[0]) 
            else:
                neigh.append(i) 
                neigh.append(i) 

        graph.update({i:neigh})
        del neigh
    
    return mask, graph

def read_file(posfile, diafile, rmax, rmin, lcylinder, rcutoff):
    with open(posfile,'r') as f:
        text = f.read()
    lines = text.splitlines()
    nparticles = int(lines[3].split()[0])
    #pos = np.zeros((nparticles,3))
    #types = np.zeros((nparticles),dtype='int')
    #drug_id = np.zeros((nparticles),dtype='int')
    pos = []
    types = []
    drug_id = []
    for i in range(9):
        del lines[0]
    for l, line in enumerate(lines):
        line = line.split()
        '''
        pos[l,0] = float(line[-3])
        pos[l,1] = float(line[-2])
        pos[l,2] = float(line[-1])
        drug_id[l] = int(line[1])
        types[l] = int(line[2])-1
        '''
        z = float(line[-1])
        if np.abs(z) - 5 > lcylinder:
            continue
        x = float(line[-3])
        y = float(line[-2])
        pos.append([x,y,z])
        drug_id.append(int(line[1]))
        types.append(int(line[2])-1)
    pos = np.array(pos)
    drug_id = np.array(drug_id,dtype='int')
    types = np.array(types,dtype='int')

    ### here we can only consider rods that are at least
    ### rmin from center (save on comp intensity/memory)
    rdis = np.sum(np.square(pos[:,:2]),axis=1)
    mask = rdis >= rmin*rmin
    pos = pos[mask]
    types = types[mask]
    drug_id = drug_id[mask]

    ## ignore rods beyond rmax from center
#    '''
    rdis = np.sum(np.square(pos[:,:2]),axis=1)
    mask = rdis <= (rmax+6)*(rmax+6)
    pos = pos[mask]
    types = types[mask]
    drug_id = drug_id[mask]
#    '''

    # read in diameters, enables dispersity
    with open(diafile,'r') as f:
        text = f.read()
    lines = text.splitlines()
    diameters = []
    for l, line in enumerate(lines):
        diameters.append(float(line.split()[0]))
    diameters = np.array(diameters)
    diameters = diameters[types]
    assert(len(diameters) == len(pos))

    # add connector 'particles' between different drug rods
    # and drug rods and cylinder surface
    # to ensure they are considered linked
    connectors = []
    connectors_surf = []
    dia_con = []
    dia_con_surf = []
    max_dia = np.max(diameters)
    tree = spatial.KDTree(pos)
    rcutoff2 = rcutoff*rcutoff
    rmax2 = rmax*rmax
    for i, ind in enumerate(pos):
        # check if drug particle is close to surface
        vector = pos[i,:2]/linalg.norm(pos[i,:2])
        rtemp = np.sum(np.square(pos[i,:2]+vector*diameters[i]/2.+vector*rcutoff))
        if rtemp >= rmax2:
            temp11 = vector*diameters[i]/2. + vector*rcutoff/2.
            p1 = pos[i] + np.append(temp11,0)
            #p1 = pos[i] + vector*diameters[i]/2. + vector*rcutoff/2.
            connectors_surf.append(p1)
            dia_con_surf.append(diameters[i])

        # connect inter-drug crystals together
        distance = (diameters[i]+max_dia)/2.
        # get each point's neighbors
        neigh = tree.query_ball_point(ind,distance+rcutoff)
        # only consider inter-crystal points
        for j, element in enumerate(neigh):
            if drug_id[i] != drug_id[element]:
                vector = pos[element] - pos[i]
                vector = vector/linalg.norm(vector)
                p1 = pos[i] + vector*diameters[i]/2.
                p2 = pos[element] - vector*diameters[element]/2.
                temp_pos = (p1+p2)/2.
                connectors.append(temp_pos)
                # set size for the connector
                dia_con.append((diameters[i]+diameters[element])/2.)
    # dont want both pairs of connectors
    connectors = np.unique(connectors,axis=0)
    # ensure that at least 1 particle is connected to cylinder surface
    #assert len(connectors_surf)>0, 'no particles connected to implant surface'
    #if len(connectors_surf)>0:
    #    connectors = np.append(connectors,connectors_surf,axis=0)
    print('number of surface contacts',len(connectors_surf))
    return pos, diameters, connectors, connectors_surf, dia_con, dia_con_surf
    #return pos, diameters, connectors

def main():
    ## pre-process the lammps output file to prepare for FD code

    #### constants one can change, unit is based on LAMMPS unit
    ## drug crystal density
    crys_den = 1.59 #g/cm^3
    # convert units to mg/um^3 - 1g=1000mg & 1cm = 10000um
    crys_den = crys_den / (10000)**3 # now g/um^3
    crys_den = crys_den * 1000 # now mg/um^3
    ## sets spacing between grid points
    grid_res = 5 # um 
    ## sets max separation between drug crystals to still consider
    ## them connected
    drug_connected_cutoff = 1 # um 
    drug_connected_cutoff = np.max([grid_res,drug_connected_cutoff])
    ## cylinder implant dimensions
    r_cylinder = 200
    ## annulus radius enables a user to only consider drug crystals
    ## at least ann_radius from cylinder center. Should save on
    ## computational intensity and memory usage especially if fitting
    ## drug release curve to an expression (only need early time)
    ann_radius = 0
    # length of cylinder
    l_cylinder = 10

    # read in lammps file to get positions of each sphere in rod
    # and diameter of each sphere
    posfile = 'output.txt'
    diafile = 'diameters.txt'
    if len(sys.argv) > 1:
        posfile = sys.argv[1]
        if len(sys.argv) > 2:
            diafile = sys.argv[2]
    pos, diameters, connectors, connectors_surf, dia_con, dia_con_surf = read_file(posfile,diafile,r_cylinder,ann_radius,l_cylinder,drug_connected_cutoff)
   
    ## ideally check for drugs with no path to surface and remove all of them
    ## remove non-connected drugs; delta_drug_distance sets drugs as connected
    ## if drugs are at most that distance away
    #positions = remove_nonconnect(positions, distance, delta_drug_distance)
    ### don't have time to do that, instead implemented annulus ability to
    ### help as well

    # generate grid
    xlow = -r_cylinder-10
    xhigh = r_cylinder+10
    ylow = -r_cylinder-10
    yhigh = r_cylinder+10
    zlow = -l_cylinder
    zhigh = l_cylinder
    xpoint = int(np.rint((xhigh-xlow)/grid_res)) + 1
    ypoint = int(np.rint((yhigh-ylow)/grid_res)) + 1
    zpoint = int(np.rint((zhigh-zlow)/grid_res)) + 1
    print('starting grid point count',xpoint*ypoint*zpoint)
    x1,y1,z1 = np.meshgrid(np.linspace(xlow,xhigh,xpoint),np.linspace(ylow,yhigh,ypoint),np.linspace(zlow,zhigh,zpoint),indexing='xy')
    x1 = x1.reshape(xpoint*ypoint*zpoint)
    y1 = y1.reshape(xpoint*ypoint*zpoint)
    z1 = z1.reshape(xpoint*ypoint*zpoint)
    gridpos = np.append(np.append(x1,y1),z1)
    gridpos = gridpos.reshape((3,xpoint*ypoint*zpoint)).T

    ### Ignore grid points that are within ann_radius
    ### since they won't have drug crystals
    rdis = np.sum(np.square(gridpos[:,:2]),axis=1)
    mask = rdis >= ann_radius*ann_radius
    gridpos = gridpos[mask]
    #print('updated grid points',len(gridpos))
   
    ## likewise only consider gridpoints that are reasonably close
    ## to implant cylinder, over estimates for safety
    rdis = np.sum(np.square(gridpos[:,:2]),axis=1)
    mask = rdis <= (r_cylinder+grid_res*2)*(r_cylinder+grid_res*2)
    gridpos = gridpos[mask]
    print('updated grid points',len(gridpos))
   
    # for each particle in pos, determine grid points to represent it
    # ensure that 'adjacent' particles have connecting grid points
    mask, graph = select_grid(pos, diameters, connectors, connectors_surf, dia_con, dia_con_surf, gridpos, drug_connected_cutoff, grid_res)

    # remove grid points that have no drug crystal and dont connect
    # drug crystals
    gridpos= gridpos[mask>0]
    # use mask to set drug crystal conc 
    # (dont put drug in connection grid points)
    mask = mask[mask>0]
    drug_conc = np.where(mask == 2, 1, 0)
    print('final number of grid points',len(gridpos))
    sys.stdout.flush()

    '''
    # graph grid
    fig,ax = plt.subplots()
    for k in range(len(drug_conc)):
        if gridpos[k,2] != 0:
            continue
        if drug_conc[k] > 0:
            ax.scatter(gridpos[k,0],gridpos[k,1],color='red',marker='.')
        else:
            ax.scatter(gridpos[k,0],gridpos[k,1],color='b',marker='.')
    #ax.set_xlim([-1,4])
    #ax.set_ylim([-1,4])
    fig.savefig('grid.png')
    plt.close()
    #'''

    # determine furthest grid points radially as the boundary grid points
    rdis = np.sum(np.square(gridpos[:,:2]),axis=1)
    mask = rdis >= (r_cylinder-grid_res)**2
    BC = np.where(mask,1,0)
    # find which drug crystals should be allowed to dissolve immediately
    allow_dissolution = []
    prog = int(len(BC)/10)
    for i, element in enumerate(BC):
        if i%prog == 0:
            print('progress',i/len(BC))
            sys.stdout.flush()
        # only consider grid near edge
        if element == 0:
            continue
        if drug_conc[i] >0:
            allow_dissolution.append(i)
            # remove from BC
            BC[i] = 0
            continue
        neigh = [i]
        counter = 0
        # something must be indicated to dissolve
        while counter < 1:
            spots = []
            for k in range(len(neigh)):
                for j, ind2 in enumerate(graph[neigh[k]]):
                    if drug_conc[ind2] >0:
                        allow_dissolution.append(ind2)
                        counter += 1
                    else:
                        spots.append(ind2)
            neigh = np.copy(spots)

    # graph release at time and cumulative    
    '''
    fig,ax = plt.subplots()
    for k in range(len(drug_conc)):
        if gridpos[k,2] != 0:
            continue
        if BC[k] > 0:
            ax.scatter(gridpos[k,0],gridpos[k,1],color='blue',marker='.')
        elif k in allow_dissolution:
            ax.scatter(gridpos[k,0],gridpos[k,1],color='r',marker='.')
    #ax.set_xlim([-1,4])
    #ax.set_ylim([-1,4])
    fig.savefig('grid-end.png')
    plt.close()
    '''

    ### here adjust drug_conc using drug crystal density and grid volume
    drug_conc = drug_conc * crys_den 

    # write file that has all grid positions and drug conc initially
    f1 = open('initial-grid-information.txt','w')
    f2 = open('fd-information.txt','w')
    f1.write("#Grid_point's X Y Z conc_crystal conc_dissolved BC allow_dissolve\n")
    f2.write("# conc_crystal conc_dissolved BC allow_dissolve\n")
    for i in range(len(drug_conc)):
        if i in allow_dissolution:
            temp = 1
        else:
            if BC[i]>0:
                temp = 1
            else:
                temp = 0
        f1.write('{} {} {} {} {} {} {}\n'.format(gridpos[i,0],gridpos[i,1],gridpos[i,2],drug_conc[i],0,BC[i],temp))
        f2.write('{} {} {} {}\n'.format(drug_conc[i],0,BC[i],temp))
    f1.close()
    f2.close()
    
    f1 = open('dictionary-information.txt','w')
    f1.write("#Grid_ID all_connected_neighbors\n")
    for i in range(len(drug_conc)):
        temp = '{} '.format(i)
        neigh = graph[i]
        for j in range(len(neigh)):
            temp += '{} '.format(neigh[j])
        temp += '\n'
        f1.write(temp)
    f1.close()

    f1 = open('fdtime.txt','w')
    f1.write('0 {}\n'.format(grid_res))
    f1.close()
    print('total number nodes dissolving',len(allow_dissolution)+len(BC))
    
if __name__ == '__main__':
    main()

