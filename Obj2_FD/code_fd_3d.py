## 3D finite-difference method written by C. Heil; UDel Spring 2022
import sys
import numpy as np
import time
import multiprocessing as mp

def boundary_conditions(conc, empty):
    ## last column at c = 0.0
    release = np.sum(conc[empty])
    conc[empty] = 0.0
    return conc, release

def update_dissolve(dissolve,neigh):
    #account_for connectors that arent drug crystals?
    return np.unique(np.append(dissolve,neigh))

def crystal_update(ccrystal,cdiss):
    global K, cs, dt, dx
    # ignore fully dissolved crystals
    if ccrystal == 0:
        # check if it is a connector point and let neighs dissolve
        if cdiss == 0:
            return 0, 0, True
        else:
            return 0, 0, False
    
    # define dissolution as K * SA * (cs - dissolved_conc) * dt / V
    # for simplicity assume SA is one face of grid, dx**2. cancel with V
    deltac = K * (cs - cdiss) * dt / dx
    assert(deltac >= 0)
    # only can dissolve up to crystal size ccrystal
    if deltac >= ccrystal:
        deltac = ccrystal
        # signal that neighbor crystals can start dissolving
        return deltac, 0, True
    else:
        return deltac, ccrystal-deltac, False

def conc_update(c0,c_array):
    global D, dt, dx
    # currently assumes constant,equal spacing in x, y, and z
    dim = 3*2
    dc = D * (np.sum(c_array) - dim * c0) * dt /(dx*dx)
    assert c0+dc >= 0
    return c0 + dc

def multiprocess_fd(node, c_cry, c_dis, c_dis_neigh):
    global cs
    deltaconc, cry_new, flag = crystal_update(c_cry, c_dis)
    dis_new = conc_update(c_dis, c_dis_neigh) + deltaconc
    assert(dis_new<=cs)
    dissolve_new = -1
    if flag:
        dissolve_new = node
    return [cry_new, dis_new, dissolve_new]

def run_fd(graph,conc_dis,conc_cry,starttime,runtime,BC,dissolving,n_cores):
    stimer = time.time()
    release_prof = []
    nodes = len(conc_dis)
    releasec = 0
    for timestep in range(starttime,runtime):
        # only consider nodes that have started/finished dissolution
        # this is slowest point, use multiprocessing to try to speed up
        mp_conc_cry = conc_cry[dissolving]
        mp_conc_dis = conc_dis[dissolving]
        mp_conc_neigh = [conc_dis[graph[node]] for node in dissolving]
        pool = mp.Pool(n_cores)
        one = pool.starmap(multiprocess_fd, zip(dissolving,mp_conc_cry,mp_conc_dis,mp_conc_neigh))
        pool.close()
        pool.join()
        one = np.array(one,dtype='object')
        conc_cry[dissolving] = np.array(one[:,0])
        conc_dis[dissolving] = np.array(one[:,1])
        dissolving_new = np.array(one[:,2])
        mask = dissolving_new>0
        if np.sum(mask) > 0:
            temp = dissolving_new[mask]
            dissolving = update_dissolve(dissolving,[graph[node] for node in temp])
            
        # update conc with conc_new after applying BC
        conc_dis, release = boundary_conditions(conc_dis,BC)
        releasec += release
    
        # save every 10000 dt
        if timestep%4999== 0:
            if timestep == 0:
                continue
            ftimer = time.time()
            print('checkpoint in',(ftimer-stimer)/60,'min')
            stimer = ftimer
            f1 = open('fd-information.txt','w')
            f1.write('# conc_crystal conc_dissolved BC allow_dissolve\n')
            for index1 in range(len(conc_dis)):
                temp = 0
                temp2 = 0
                if index1 in dissolving:
                    temp = 1
                if index1 in BC:
                    temp2 = index1
                f1.write('{} {} {} {}\n'.format(conc_cry[index1],conc_dis[index1],temp2,temp))
            f1.close()
            f1 = open('fdtime.txt','w')
            f1.write('{} {}\n'.format(timestep, dx))
            f1.close()
            f1 = open('fd-conc-release.txt','a')
            f1.write('{} {}\n'.format(timestep,releasec))
            f1.close()
            print('progress',timestep/runtime)
            print('release',releasec)
            release_prof.append(releasec)
            releasec = 0
            sys.stdout.flush()

    return releasec

def read_files(fdfile,dictfile, fdtime):
    conc_dis = []
    conc_cry = []
    BC = []
    dissolving = []
    with open(fdfile,'r') as f:
        text = f.read()
    lines = text.splitlines()
    del lines[0]
    for l,line in enumerate(lines):
        line = line.split()
        conc_cry.append(float(line[0]))
        conc_dis.append(float(line[1]))
        if float(line[2])>0:
            BC.append(l)
            dissolving.append(l)
        elif float(line[3])>0:
            dissolving.append(l)
   
    conc_cry = np.array(conc_cry)
    conc_dis = np.array(conc_dis)
    BC = np.array(BC)
    dissolving = np.array(dissolving)
     
    graph = {}
    with open(dictfile,'r') as f:
        text = f.read()
    lines = text.splitlines()
    del lines[0]
    for l,line in enumerate(lines):
        line = line.split()
        temp = int(line[0])
        neigh = []
        for k in range(1,7):
            neigh.append(int(line[k]))
        graph.update({temp:neigh})
    
    with open(fdtime,'r') as f:
        text = f.read()
    lines = text.splitlines()
    starttime = int(lines[0].split()[0])
    grid_res = float(lines[0].split()[1])
            
    return conc_dis, conc_cry, BC, dissolving, graph, starttime, grid_res

def main():
    global D, K, cs, dt, dx
    ### these are the constants one can adapt
    ### make sure units are correct and match pre-process
    # D from Mol. pharmaceutics 2018 15, 1488-1494 (Pio di Cagno, et al)
    D = 6*10**-6 #cm^2/s
    # convert to um (our unit of length) - 1cm = 10000um
    # K estimated from Journal of pharmaceutical sciences 2016 vol 105 issue 9, 2685-2697 (Shekunov & Montgomery) and journal of pharmaceutical sciences 1965 vol 54 issue 11 1651-1653 (Hamlin, Northam, & Wagner)
    K = 6*10**-4 # cm/s
    # convert to um (our unit of length) - 1cm = 10000um
    D *= (10000)**2
    K *= 10000
    # info given
    cs = 1 #mg/cm^3
    # convert to um (our unit of length) - 1cm = 10000um
    cs = cs / (10000.)**3 # now mg/um^3
    # dt should be selected to be as large as possible
    # without causing numerical issues
    dt = 2.75*10**(-4)
#    dt = 0.000005
    # set desired length of run in unit (s)
    # this can be increased after running
    endtime = 864000 # s -- 10 days
    runtime = int(np.rint(endtime/dt))
    print('number of steps to sim 10 days',runtime)
    # set the number of cores (intra-node only) to parallelize over
    n_cores = 64
    
    ## for 3d case, read in grid and conc values + graph 
    fdfile = 'fd-information.txt'
    dictfile = 'dictionary-information.txt'
    fdtime = 'fdtime.txt'
    if len(sys.argv) > 1:
        fdfile = sys.argv[1]
        if len(sys.argv) > 2:
            dictfile = sys.argv[2]
    conc_dis, conc_cry, BC, dissolving, graph, starttime, grid_res = read_files(fdfile,dictfile,fdtime)

    dx = grid_res
    # preemptive check
    conc, release = boundary_conditions(conc_dis,BC)
    assert np.sum(release) < 1e5
    conc, release = boundary_conditions(conc_cry,BC)
    assert np.sum(release) < 1e5

    release = run_fd(graph,conc_dis,conc_cry,starttime,runtime,BC,dissolving,n_cores)

    f1 = open('simdone.txt','w')
    f1.write('finished FD\n')
    f1.close()

    # release is given in mg
    
    # graph release at time and cumulative    
    ### note may need to see saved file depending on length of run
    fig,ax = plt.subplots()
    for k in range(len(release)):
        ax.scatter(k,release[k],color='red')
    fig.savefig('release-instant.png')
    plt.close()
    fig,ax = plt.subplots()
    for k in range(len(release)):
        ax.scatter(k,np.sum(release[:k]),color='red')
    fig.savefig('release-cumulative.png')
    plt.close()

D = 0
K = 0
cs = 0
dt = 0
dx = 0
if __name__ == '__main__':
    main()
