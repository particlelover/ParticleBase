import sys
import random
import math
import ctypes
import cudaParticles

def createInitialState(particles):
    G1 = cudaParticles.globalTable()

    ## units used in this script is
    ## [g][cm][s]
    R0 = 0.50
    lunit = 2 * R0
    L1 = 60.0
    L2 = 60.0
    L3 = 10.0
    L4 = 50.0
    L5 = 20.0

    cell = cudaParticles.new_realArray(6)
    cudaParticles.realArray_setitem(cell, 0, 0.0)
    cudaParticles.realArray_setitem(cell, 1, L1)
    cudaParticles.realArray_setitem(cell, 2, 0.0)
    cudaParticles.realArray_setitem(cell, 3, L2)
    cudaParticles.realArray_setitem(cell, 4, 0.0)
    cudaParticles.realArray_setitem(cell, 5, L4)

    WeighFe = 7.874 * 4.0 / 3.0 * math.pi * R0*R0*R0

    # border
    lsize1 = int(L1 / lunit + 1)
    lsize2 = int(L2 / lunit + 1)
    lsize3 = int(L3 / lunit + 1)

    carray3_t = ctypes.c_double * 3

    for k in range(lsize3):
        for i in range(lsize1):
            for j in range(lsize2):

                if ( ( (i<1) or (lsize1-2<i) or (j<1) or (lsize2-2<j) )
                     or
                     (k==0)
                ):
                    pb = cudaParticles.ParticleBase()

                    pb_r = carray3_t.from_address(int(pb.r))
                    pb_v = carray3_t.from_address(int(pb.v))
                    pb_a = carray3_t.from_address(int(pb.a))
                    if ( (0<i) and (i<lsize1-1) and (0<j) and (j<lsize2-1) ):
                        pb_r[0] = i*lunit + random.gauss(0.0, R0/100.0)
                        pb_r[1] = j*lunit + random.gauss(0.0, R0/100.0)
                    else:
                        pb_r[0] = i*lunit
                        pb_r[1] = j*lunit

                    pb_r[2] = k*lunit
                    pb.m = WeighFe
                    pb_v[0] = pb_v[1] = pb_v[2] = 0.0
                    pb_a[0] = pb_a[1] = pb_a[2] = 0.0
                    pb.isFixed = True
                    pb.type = 0
                    G1.push_back(pb)

    N1=G1.size()
    print >> sys.stderr, "N=", N1


    # moving
    P1 = int(L5 / lunit / 1.3)
    P2 = int(L4 / lunit - 2)

    for k in range(P2):
        for i in range(P1):
            for j in range(P1):
                pb = cudaParticles.ParticleBase()
                pb_r = carray3_t.from_address(int(pb.r))
                pb_v = carray3_t.from_address(int(pb.v))
                pb_a = carray3_t.from_address(int(pb.a))
                pb_r[0] = (i*1.3)*lunit + random.gauss(0.0, R0/10.0) + L5
                pb_r[1] = (j*1.3)*lunit + random.gauss(0.0, R0/10.0) + L5
                pb_r[2] = (k+1  )*lunit + lunit/3
                pb.m = WeighFe
                pb_v[0] = pb_v[1] = pb_v[2] = 0.0
                pb_a[0] = pb_a[1] = pb_a[2] = 0.0
                pb.isFixed = False
                pb.type = 1
                G1.push_back(pb)

    N = G1.size()
    print >> sys.stderr, "N=", N


    ndev = particles.nDevices()
    particles.setup()

    for i in range(ndev):
        particles.setGPU(i)

        particles[i].setup(N)
        particles[i].setCell(cell)

        particles[i]._import(G1)

        particles[i].setDEMProperties(2.11e10, 0.40, 0.29, 0.10332, 0.10, R0)
        particles[i].setInertia(R0)
        particles[i].setupCutoffBlock(R0*2/math.sqrt(3.0)*0.9, False)

        # putTMPselected
        particles[i].setupSelectedTMP(N1, N-N1, 0, N1)


        particles[i].checkPidRange(i, ndev, N1, N)

    particles[0].timestep = 0
    particles[0].putUnSelected("dump.DEM5pybox")



if __name__ == "__main__":
    particles = cudaParticles.CudaParticleDEM()
    ndev = particles.nDevices()

    if len(sys.argv) == 2:
        print "reading serialization file ", sys.argv[1]
        particles.setup()

        particles.readSerialization(sys.argv[1])
    else:
        createInitialState(particles)

        for i in range(ndev):
            particles.setGPU(i)
            particles[i].calcBlockID()


    particles[0].getSelectedTypeID()
    particles[0].getSelectedPosition()
    particles[0].putTMP()
    particles[0].waitPutTMP()


    stepmax  = 1.50
    intaval  = 0.005
    initstep = particles[0].timestep
    initDeltaT = 0.000008
    ulim = 0.01 * 4
    llim = ulim / 16.0
    R0 = 0.50
    res = cudaParticles.VectorInt(ndev)

    param_g = 9.8e2

    thistime = cudaParticles.AdaptiveTimeDEM()
    nextoutput = intaval
    thistime.init(initDeltaT)
    thistime.statOutput = True

    thistime.PrintStat(0)
    print >> sys.stderr, "End Time: ", stepmax

    for i in range(ndev):
        particles.setGPU(i)
        particles[i].selectBlocks()
        particles[i].calcVinit(initDeltaT)

    j = 0

    while thistime() <stepmax:
        if thistime.isRollbacking(): print >> sys.stderr, "now rollbacking:",

        if j % 50 == 0: print >> sys.stderr, j,

        for i in range(ndev):
            res[i] = 0
            particles.setGPU(i)
            particles[i].calcBlockID()

            particles[i].selectBlocks()
            particles[i].calcForce(thistime.currentDeltaT())

        if ndev > 1:
            particles.exchangeForceSelected(0)
            particles.exchangeForceSelected(2)

        for i in range(ndev):
            particles.setGPU(i)
            particles[i].calcAcceleration()
            particles[i].addAccelerationZ(-param_g)
            particles[i].TimeEvolution(thistime.currentDeltaT())
            res[i] = particles[i].inspectVelocity((2*R0)/thistime.currentDeltaT(), ulim, llim)

        j = thistime.Progress(particles, res, j)


        for i in range(ndev):
            particles.setGPU(i)
            particles[i].treatRefrectCondition()

        if thistime() >= nextoutput:
            print >> sys.stderr, "({})".format(thistime())
            nextoutput = nextoutput + intaval
            particles[0].timestep = j+1+initstep
            particles[0].getSelectedPosition()
            particles[0].putTMP()

    particles[0].waitPutTMP()
    thistime.PrintStat(j)

    particles.writeSerialization("DEM5donepy")
    print "Done."
