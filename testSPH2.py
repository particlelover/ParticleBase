import sys
import cudaParticles


if __name__ == "__main__":
    particles = cudaParticles.CudaParticleSPH_NS()
    ndev = particles.nDevices()

    if len(sys.argv) == 2:
        print "reading serialization file ", sys.argv[1]
        particles.setup()

        particles.readSerialization(sys.argv[1])
    else:
        print "not supported yet"
        sys.exit()


    for i in range(ndev):
        particles[i].setBlockRange(particles[i].numBlocks(), ndev, i)


    particles[0].getTypeID()
    particles[0].getPosition()
    particles[0].putTMP()


    deltaT   = 0.000050
    stepmax  = int(0.50 / deltaT)
    intaval  = int(0.00100 / deltaT)
    initstep = particles[0].timestep


    for i in range(ndev):
        particles.setGPU(i)
        particles[i].calcVinit(deltaT)


    for j in range(stepmax):
        print >> sys.stderr, j,
        for i in range(ndev):
            particles.setGPU(i)
            particles[i].calcBlockID()

            particles[i].calcKernels()
            particles[i].calcDensity()

            particles[i].selectBlocks()
            particles[i].setSelectedRange(particles[i].numSelectedBlocks(), ndev, i)

            particles[i].calcForce()
            particles[i].calcAcceleration()

        if ndev>1:
            particles.exchangeAccelerations()

        for i in range(ndev):
            particles.setGPU(i)
            particles[i].addAccelerationZ(-9.8e2)
            particles[i].TimeEvolution(deltaT)
            particles[i].treatAbsoluteCondition()

        if (j+1)%intaval==0:
            particles[0].timestep = j+1+initstep
            particles[0].getPosition()
            particles[0].putTMP()


    particles.writeSerialization(sys.argv[1])
    print "Done."
