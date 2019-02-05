# cudaParticleBase

Basic classes for the particle style simulations such as
Molecular Dynamics, Smoothed Particle Hydrodynamics,
Distinct Element Method, by using CUDA.

Time evolutions of points, calculation of forces of i-j pairs,
pair potentials are defined in CUDA kernel functions.
CUBLAS is also used.

Auto changing the time step of Euler scheme in DEM simulation is implemented.

Support serialization of particles object on GPU by boost.

OpenMP and pthread are also used.

