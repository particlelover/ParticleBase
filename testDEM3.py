import sys
import cudaParticles


if __name__ == "__main__":
    particles = cudaParticles.CudaParticleDEM()
    ndev = particles.nDevices()

    if len(sys.argv) == 2:
        print "reading serialization file ", sys.argv[1]
        particles.setup()

        particles.readSerialization(sys.argv[1])
    else:
        print >> sys.stderr, "not supported yet"
        sys.exit()



    ## calculate densities for the first output
    particles[0].calcBlockID()

    particles[0].getSelectedTypeID()
    particles[0].timestep = 0
    particles[0].getSelectedPosition()
    particles[0].putTMP()


    deltaT   = 0.0000015
    stepmax  = int(1.50 / deltaT)
    intaval  = int(0.005 / deltaT)
    initstep = particles[0].timestep

    particles[0].selectBlocks()
    particles[0].calcVinit(deltaT)

    for j in range(stepmax):
        print >> sys.stderr, j,
        particles[0].calcBlockID()


        particles[0].selectBlocks()

        particles[0].calcForce(deltaT)
        particles[0].calcAcceleration()
        particles[0].addAccelerationZ(-9.8e2)
        particles[0].TimeEvolution(deltaT)
        #particles[0].treatAbsoluteCondition();

        if (j+1)%intaval==0:
            particles[0].timestep = j+1+initstep
            particles[0].getPosition()
            particles[0].putTMP()


    particles.writeSerialization(sys.argv[1])
    print "Done."
