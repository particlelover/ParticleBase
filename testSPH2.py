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

    
    B0 = particles[0].numBlocks()
    B1 = B0 / ndev
    B2 = 0
    for i in range(ndev):
        particles[i].setMyBlock(B2, B1)
        print "set block range for GPU ", i, ": ", B2, " to ", B2+B1
        B2 += B1

    if (B1*ndev)!=B0:
        particles[ndev-1].setMyBlock(B2-B1, B1+(B0-(B1*ndev)))
        print "correct block range for GPU ", ndev-1, ": ", B2-B1, " to ", B2+(B0-(B1*ndev))


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


        _k=0
        for _n in range(ndev):
            particles[_n].myOffsetSelected = _k
            _k += particles[_n].myBlockSelected
        particles[ndev-1].myBlockSelected += (particles[ndev-1].numSelectedBlocks()-_k)
            

        for i in range(ndev):
            particles.setGPU(i)
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
