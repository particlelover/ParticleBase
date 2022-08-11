#if !defined(KERNELFUNCS)
#define KERNELFUNCS
#include <vector>
#include <stdint.h>


typedef REAL  real;

#define R2(x) x##2
#define RR(x) R2(x)
#define REAL2 RR(REAL)



/** clear array on GPU with a size num
 *
 * @param r array on GPU
 * @param num size of array
 * @param val cleared by this value (default 0)
 */
__global__ void clearArray_F4(float4 *r, uint32_t num, float4 val={0.0, 0.0, 0.0, 0.0});

/** add a value to array
 *
 * @param r array on GPU
 * @param num size of array
 * @param val value to add
 */
__global__ void addArray_F4(float4 *r, float4 val, uint32_t num);

/** add an array to another array
 *
 * @param r array on GPU
 * @param num size of array
 * @param val value to add
 */
__global__ void addArray_F4(float4 *r, float4 *val, uint32_t num);

/** calc reciprocal of array r[] to 1/r[]
 *
 * @param r array on GPU
 * @param rinv  results array on GPU
 * @param num size of array
 */
__global__ void calcReciproc(real *r, real *rinv, uint32_t num);

/** calc A[i] *= B[i]
 *
 * @param A array on GPU
 * @param B array on GPU
 * @param num size of array
 */
__global__ void multiplies(real *A, real *B, uint32_t num);

/** calc Acceleration from Force by a = F/m
 *
 * @param a array for results acceleration
 * @param minv  array for 1/mass
 * @param F array for force
 * @param N number of particles (array size is 3N)
 */
__global__ void calcA_F4(float4 *a, real *minv, float4 *F, uint32_t N);

/** apply periodic boundary condition to particles
 *
 * @param r array for results position
 * @param c0  lower boundary of the cell
 * @param c1  upper boundary of the cell
 * @param N number of particles (array size is 3N)
 */
__global__ void applyPeriodicCondition_F4(float4 *r, float3 c0, float3 c1, uint32_t N);

/** treatments of the boundary condition
 *
 * return particles outside of the simulation cell back into cell region
 *
 * @param r array for results position
 * @param c0  lower boundary of the cell
 * @param c1  upper boundary of the cell
 * @param N number of particles (array size is 3N)
 */
__global__ void treatAbsoluteBoundary_F4(float4 *r, float4 c0, float4 c1, uint32_t N);

/** treatments of the boundary condition
 *
 * return particles outside of the simulation cell back into cell region
 *
 * @param r array for reference positions
 * @param v array for results veklocities
 * @param c0  lower boundary of the cell
 * @param c1  upper boundary of the cell
 * @param N number of particles (array size is 3N)
 */
__global__ void treatRefrectBoundary_F4(float4 *r, float4 *v, float4 c0, float4 c1, uint32_t N);

/** do propagation with Euler method
 *
 * @param r array for results position
 * @param dt  delta t
 * @param v array for result velocity
 * @param a array for acceleration
 * @param move  array for move flag
 * @param N number of particles (array size is 3N)
 */
__global__ void propagateEuler_F4(float4 *r, real dt, float4 *v, float4 *a, char *move, uint32_t N);

/** inspect velocity
 *
 * @param v array for result velocity
 * @param N number of particles (array size is 3N)
 * @param vlim  maximum velocity
 * @param tmp
 * @param thresh threshold upper/lower limit for the v/vlim
 * @param DEBUG if true, obtain all vratio
 */
__global__ void inspectV_F4(float4 *v, uint32_t N, uint32_t vlim, float4 *tmp, float2 thresh, bool debug=false);

/** calculates v(0-dt/2) from v(0) and F(0) for Leap Frog method
 *
 * @param v array for result velocity
 * @param dt_2  delta t / 2
 * @param F array for force
 * @param minv  array for 1/mass
 * @param N number of particles (array size is 3N)
 */
__global__ void calcLFVinit_F4(float4 *v, real dt_2, float4 *F, real *minv, uint32_t N);

/** do propagation with Leap Frog
 *
 * @param r array for results position
 * @param dt  delta t
 * @param v array for result velocity
 * @param a array for acceleration
 * @param minv  array for 1/mass
 * @param move  array for move flag
 * @param N number of particles (array size is 3N)
 */
__global__ void propagateLeapFrog_F4(float4 *r, real dt, float4 *v, float4 *a, real *minv, char *move, uint32_t N);


/** rollback the position and velocity by Leap Frog
 *
 */
__global__ void rollbackLeapFrog_F4(float4 *r, real dt, float4 *v, float4 *a, real *minv, char *move, uint32_t N);


/** do propagation with Velocity-Verlet (skipping acceleration)
 *
 * @param r array for results position
 * @param dt  delta t
 * @param v array for result velocity
 * @param F array for Force
 * @param Fold  array for previous Force
 * @param minv  array for 1/mass
 * @param N number of particles (array size is 3N)
 */
__global__ void propagateVelocityVerlet_F4(float4 *r, real dt, float4 *v, float4 *F, float4 *Fold, real *minv, uint32_t N);

/** calculate constant term at velocity update in Velocity-Verlet with
 * Gaussian Thermostat; called from cudaParticleVV::timeEvolutonGaussianThermo()
 *
 * A(t) = v(t-dt)+(F(t)+F(t-dt))dt/2m - v(t-dt)xi(t-dt)dt/2
 * v(t) is also updated by oridinary velocity-verlet
 *
 * @param A array for results (tmp81N)
 * @param dt  delta t
 * @param vd array for result velocity
 * @param F array for Force
 * @param Fold  array for previous Force
 * @param minv  array for 1/mass
 * @param N number of particles (array size is 3N)
 * @param xi  \xi at (t-dt)
 */
__global__ void calcGaussianThermoA1_F4(double *A, real dt, double *vd, float4 *F, float4 *Fold, real *minv, uint32_t N, double xi);


/** do propagation with Velocity-Verlet + Gaussian Thermostat
 * (update velocity by Newton-Raphson iteration)
 *
 * @param A array for constant term A1
 * @param dt  delta t
 * @param vd array for result velocity
 * @param F array for Force
 * @param m array for mass
 * @param N number of particles (array size is 3N)
 * @param xi  \xi at previous step
 * @param mv2inv  \f$ 1 / (\sum mv^2) \f$ at previous step
 */
__global__ void calcGaussianThermoFoverF_F4(double *A, real dt, double *vd, float4 *F, real *m, double *tmp3N, uint32_t N, double xi, real mv2inv);

/** do propagation with Velocity-Verlet + Gaussian Thermostat
 * (update position)
 *
 * @param r array for results position
 * @param dt  delta t
 * @param v array for result velocity
 * @param F array for Force
 * @param Fold  array for previous Force
 * @param minv  array for 1/mass
 * @param N number of particles (array size is 3N)
 * @param xi  \xi at (t)
 */
__global__ void propagateVelocityVerletGaussianThermo_F4(float4 *r, real dt, double *vd, float4 *F, float4 *Fold, real *minv, uint32_t N, double xi);

/** inner product of two vector v1 and v2;
 * @return \sum v1_i.x*v2_i.x + v1_i.y*v2_i.y + v1_i.z*v2_i.z
 *
 * @param v1
 * @param v2
 * @param N length of vector
 * @param Res result
 */
__global__ void innerProduct_F4(float4 *v1, double *vd, uint32_t N, double *Res);

/** calculate the \f$ \sum m_i v_i^2 \f$ by shared memory and reduction
 *
 * @return  3N kB T
 */
__global__ void calcMV2_F4(double *vd, float* m, double *tmp3N, uint32_t N);

/** calculate the squared velocity (v^2) for each particles and store to tmp array
 *
 * @param v array for velocity
 * @param tmp3N array to store the results
 * @param N number of particles
 */
__global__ void calcV2_F4(float4 *v, float *tmp3N, uint32_t N);


/** correct Force by Gaussian Thermostat
 *
 * this correct forces by its velocities, mass and term lambda as follows
 * \f[
 *  F_i -= \lambda * m_i * v_i
 * \f]
 *
 * it is still inproper to calculate v(t) in velocity-velret with forces
 * corrected by this method
 *
 * @param v array for velocity
 * @param F array to correct
 * @param m array for mass of particles
 * @param lambda  correction term
 * @param N number of particles
 */
__global__ void correctConstTemp_F4(float4 *v, float4 *F, real *m, real lambda, uint32_t N);


/** adjust velocity of particles by calculating the mean and standard deviations
 * 
 * 
 * @param v     array for velocity
 * @param st    stride width (same with number of threads)
 * @param N     number of particles
 * @param v1    sqrt(kB * Temp / m0)
 */
__global__ void adjustVelocity_F4(float4 *v, uint32_t st, uint32_t N, real v1, float *debug=NULL);
__global__ void adjustVelocity_VD(double *vd, uint32_t st, uint32_t N, real v1, float *debug=NULL);


/** calculate forces from SPH particles by Navier-Stokes equation
 *
 * @param r array for position (size 3N)
 * @param a results array for calculated accelerations (size 3N)
 * @param typeID  array for type ID of particles (size N)
 * @param N number of particles
 * @param dW2D  2D dW/dr array for all i-j pair
 * @param rhoinv  reciprocal of mass density field (1/rho)
 * @param m mass of particles
 * @param mu  array for shear viscosity
 * @param v array for velocity field
 */
__global__ void calcF_SPH_NS(real *r, real *a, unsigned short *typeID, uint32_t N, real *dW2D, real *rhoinv, real *m, real *mu, real *v, const real K);

/** inspect density at SPH particles to treat particles on the free surface
 *
 * @param n number density field
 * @param move  array for move flag
 * @param N number of particles
 * @param R result array for the density (only R[0] contains the result)
 */
__global__ void inspectDense_x(float4 *n, char *move, uint32_t N, float4 *R);


/** calculate ID of cutoff block
 * by block_Z + block_Y * num_Z + block_X * num_Z * num_Y
 *
 * calculated block ID and its particle ID is stored in bid[i] and pid[i] for i-th particle
 *
 * @param r position of particles
 * @param bid block ID of particles
 * @param pid particle ID
 * @param N number of particles
 * @param b0  length of block for X
 * @param b1  length of the block for Y
 * @param b2  length of the block for Z
 * @param cxmin min of cell X
 * @param cymin min of cell Y
 * @param czmin min of cell Z
 * @param c0  number of the block in x
 * @param c1  number of the block in y
 * @param c2  number of the block in z
 */
__global__ void calcBID_F4(float4 *r, uint32_t *bid , uint32_t *pid, uint32_t N, uint32_t totalNumBlock,
    real b0, real b1, real b2,
    real cxmin, real cymin, real czmin,
    uint32_t c0, uint32_t c1, uint32_t c2);

/** sort bid table in parallel, pid table is aloso modified by satisfing
 * particle pid[i] is located in bid[i]
 *
 * bindex table is also calculated, which is partial sum of number of particles
 * in each blocks
 *
 * @param bid block ID of particles
 * @param pid particle ID
 * @param N number of particles
 */
__global__ void sortByBID(uint32_t *pid, uint32_t *bid, uint32_t N);

/** makes/re-calculates bindex[] from sorted bid[]
 *
 * @param bid block ID of particles
 * @param bindex  partial sum of number of particles
 * @param N number of particles
 */
__global__ void makeBindex(uint32_t *bid , uint32_t *bindex, uint32_t N);

/** first phase to sort bid table by merge sort, bubble sort with a stride #thread x #MP
 *
 * pid table is aloso modified by satisfing
 * particle pid[i] is located in bid[i]
 *
 * bindex table is also calculated, which is partial sum of number of particles
 * in each blocks
 *
 * @param bid block ID of particles
 * @param pid particle ID
 * @param N number of particles
 */
__global__ void sortByBID_M1(uint32_t *pid, uint32_t *bid, uint32_t N);

/** acclumulate tmp81N array to the target array
 *
 * @param A array for the target
 * @param A27 array with the size N*27*3
 * @param N size of target array (particle num)
 */
__global__ void reduce27_F4(float4 *A, real *A27, uint32_t N);



/** Restore an array by pid; dst[pid[i]] =src[i]
 *
 * @param dst an array (size 3N)
 * @param src
 * @param N array length
 * @param pid particle ID map
 */
__global__ void RestoreByPid_F4(float4 *dst, float4 *src, uint32_t N, uint32_t *pid);


/** calculate the Eulerian Equation of Motion (Leap-Frog method)
 *
 * atitude and inertia are fixed, and initial inertia is 2/5 r^2 M E
 *
 * \f[
 *  L_i += \Delta t * T_i \\
 *  \omega _i = \frac{1}{m_0} L_i
 * \f]
 */
__global__ void propagateEulerianRotation_F4(float4 *w, real dt, float4 *L, float4 *T, real *m0inv, char *move, uint32_t N);


/** check cutoff blocks to select blocks which have movable particles
 *
 * @param selected  1D table for the block selection
 * @param pid particle ID table (sorted by block ID)
 * @param move  move flags for each particles
 * @param bindex  index for the pid[] array for each blocks
 * @param N total number of the cutoff blocks
 */
__global__ void checkBlocks(uint32_t *selected, uint32_t *pid, char *move,
    uint32_t *bindex, uint32_t N);

/** accumulate 1D array with shared memory and reduction
 * (MPnum must be 1)
 *
 * @param selected  1D table for the block selection
 * @param N total number of the cutoff blocks
 * @param Res results transfered from GPU's device memory to host memory
 */
__global__ void accumulate_F4(uint32_t *selected, uint32_t N, float4 *Res);

__global__ void accumulate(float4 *selected, uint32_t N, float4 *Res);

/* write block ID to the selectedBlock[] table
 *
 * @param selected  1D table for the block selection
 * @param N total number of the cutoff blocks
 */
__global__ void writeBlockID(uint32_t *selected, uint32_t N);

/** cast real[] to uint32_t[] on GPU device memory
 *
 * @param r array for real (src)
 * @param l array for uint32_t (dest)
 * @param N size of arrays
 * @param pid pid[] array to convert i=pid[_i] if given
 */
__global__ void real2ulong_F4(float4 *r, uint32_t *l, uint32_t N, uint32_t *pid=NULL);

/** calculates partial sum with prefix scan
 *
 */
__global__ void partialsum(uint32_t *rowPtr, uint32_t N);

/** make list of particle j around i as colIdx[] by using Cutoff Block
 *
 * @param rowPrt row pointer for each i particle
 * @param colIdx  results array, colIdx[rowPtr[i]..rowPtr[i+1]] stores ID of particle j around i
 * @param r0  array for radius of each particles
 * @param r array for position
 * @param blockNeighbor neighbor judgement for I-J block pair
 * @param pid particle ID in sorted order
 * @param bindex  start particle num of this block
 * @param bid_by_pid  array for the Block ID
 * @param move  move/fixed particle table
 * @param N number of particles
 */
__global__ void makeJlist_WithBlock2_F4(const uint32_t *rowPtr, uint32_t *colIdx,
    const float4 *r,
    const uint32_t *blockNeighbor, const uint32_t *pid, const uint32_t *bindex,
    const uint32_t *bid_by_pid, const char *move,
    const uint32_t N, const uint32_t Nstart, const uint32_t Nend
  );

__global__ void makeJlist_WithBlock4_F4(const uint32_t *rowPtr, uint32_t *colIdx,
    const float4 *r,
    const uint32_t *blockNeighbor, const uint32_t *bindex,
    const uint32_t *bid, const char *move,
    const uint32_t N,
    const uint32_t Nstart, const uint32_t Nend
  );

__global__ void calcBID_direct_F4(const float4 *r, uint32_t *bid, uint32_t *bindex,
    const uint32_t N, const real b0, const real b1, const real b2,
    const real cxmin, const real cymin, const real czmin,
    const uint32_t blen0, const uint32_t blen1, const uint32_t blen2,
    const uint32_t CX, const uint32_t CY);

/** makes bid_by_pid table
 * @param pid pid table
 * @param bid bid table
 * @param bid_by_pid converts pid to bid
 * @param N number of particles
 */
__global__ void makeBIDbyPID(const uint32_t *pid, uint32_t *bid, uint32_t *bid_by_pid,
    const uint32_t N);

/** sort colIdx in parallel, each row for i are sorted in odd-even sort
 *
 * @param rowPtr rowPtr in CSR format
 * @param colIdx colIdx in CSR format
 * @param N number of row
 */
__global__ void sortColIdx(const uint32_t *rowPtr, uint32_t *colIdx, uint32_t N);

__global__ void succeedPrevState(uint32_t *prevRow, uint32_t *prevCol, real *prevVal,
    uint32_t *curRow, uint32_t *curCol, real *curVal,
    uint32_t N);

#endif /* KERNELFUNCS */
