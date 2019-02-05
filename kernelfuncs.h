#if !defined(KERNELFUNCS)
#define KERNELFUNCS
#include <vector>
#include <stdint.h>

typedef REAL real;

#define R2(x) x##2
#define RR(x) R2(x)
#define REAL2 RR(REAL)

/** add value to array
 *
 * @param r	array on GPU
 * @param num size of array
 * @param val	value to add
 */
__global__ void addArray(real *r, real val, uint32_t num);

/** multiply array by value
 *
 * @param r	array on GPU
 * @param num size of array
 * @param val	value to multiply
 */
__global__ void mulArray(real *r, real val, uint32_t num);

/** calc reciprocal of array r[] to 1/r[]
 *
 * @param r	array on GPU
 * @param rinv	results array on GPU
 * @param num size of array
 */
__global__ void calcReciproc(real *r, real *rinv, uint32_t num);

/** calc A[i] *= B[i]
 *
 * @param A	array on GPU
 * @param B	array on GPU
 * @param num size of array
 */
__global__ void multiplies(real *A, real *B, uint32_t num);

/** calc Acceleration from Force by a = F/m
 *
 * @param a	array for results acceleration
 * @param minv	array for 1/mass
 * @param F array for force
 * @param N number of particles (array size is 3N)
 */
__global__ void calcA(real *a, real *minv, real *F, uint32_t N);

/** apply periodic boundary condition to particles
 *
 * @param r	array for results position
 * @param c0	lower boundary of the cell
 * @param c1	upper boundary of the cell
 * @param N number of particles (array size is 3N)
 */
__global__ void applyPeriodicCondition(real *r, real c0, real c1, uint32_t N);

/** treatments of the boundary condition
 *
 * return particles outside of the simulation cell back into cell region
 *
 * @param r	array for results position
 * @param c0	lower boundary of the cell
 * @param c1	upper boundary of the cell
 * @param N number of particles (array size is 3N)
 */
__global__ void treatAbsoluteBoundary(real *r, real c0, real c1, uint32_t N);

/** do propagation with Euler method
 *
 * @param r	array for results position
 * @param dt	delta t
 * @param v	array for result velocity
 * @param a	array for acceleration
 * @param move	array for move flag
 * @param N number of particles (array size is 3N)
 */
__global__ void propagateEuler(real *r, real dt, real *v, real *a, char *move, uint32_t N);

/** inspect velocity
 *
 * @param v	array for result velocity
 * @param N	number of particles (array size is 3N)
 * @param vlim	maximum velocity	
 * @param tmp
 * @param lim_u upper limit
 * @param lim_l lower limit
 * @param DEBUG	if true, obtain all vratio
 */
__global__ void inspectV(real *v, uint32_t N, uint32_t vlim, real *tmp, real lim_u, real lim_l, bool debug = false);

/** calculates v(0-dt/2) from v(0) and F(0) for Leap Frog method
 *
 * @param v	array for result velocity
 * @param dt_2	delta t / 2
 * @param F	array for force
 * @param minv	array for 1/mass
 * @param N number of particles (array size is 3N)
 */
__global__ void calcLFVinit(real *v, real dt_2, real *F, real *minv, uint32_t N);

/** do propagation with Leap Frog
 *
 * @param r	array for results position
 * @param dt	delta t
 * @param v	array for result velocity
 * @param a	array for acceleration
 * @param minv	array for 1/mass
 * @param move	array for move flag
 * @param N number of particles (array size is 3N)
 */
__global__ void propagateLeapFrog(real *r, real dt, real *v, real *a, real *minv, char *move, uint32_t N);

/** rollback the position and velocity by Leap Frog
 *
 */
__global__ void rollbackLeapFrog(real *r, real dt, real *v, real *a, real *minv, char *move, uint32_t N);

/** do propagation with Velocity-Verlet (skipping acceleration)
 *
 * @param r	array for results position
 * @param dt	delta t
 * @param v	array for result velocity
 * @param F	array for Force
 * @param Fold	array for previous Force
 * @param minv	array for 1/mass
 * @param N number of particles (array size is 3N)
 */
__global__ void propagateVelocityVerlet(real *r, real dt, real *v, real *F, real *Fold, real *minv, uint32_t N);

/** calculate constant term at velocity update in Velocity-Verlet with
 * Gaussian Thermostat; called from cudaParticleVV::timeEvolutonGaussianThermo()
 *
 * A(t) = v(t-dt)+(F(t)+F(t-dt))dt/2m - v(t-dt)xi(t-dt)dt/2
 * v(t) is also updated by oridinary velocity-verlet
 *
 * @param A	array for results (tmp3N)
 * @param dt	delta t
 * @param v	array for result velocity
 * @param F	array for Force
 * @param Fold	array for previous Force
 * @param minv	array for 1/mass
 * @param N number of particles (array size is 3N)
 * @param xi	\xi at (t-dt)
 */
__global__ void calcGaussianThermoA1(real *A, real dt, real *v, real *F, real *Fold, real *minv, uint32_t N, real xi);

/** do propagation with Velocity-Verlet + Gaussian Thermostat
 * (update velocity by Newton-Raphson iteration)
 *
 * @param A	array for constant term A1
 * @param dt	delta t
 * @param v	array for result velocity
 * @param F	array for Force
 * @param m	array for mass
 * @param N number of particles (array size is 3N)
 * @param xi	\xi at previous step
 * @param mv2inv	\f$ 1 / (\sum mv^2) \f$ at previous step
 */
__global__ void calcGaussianThermoFoverF(real *A, real dt, real *v, real *F, real *m, real *tmp3N, uint32_t N, real xi, real mv2inv);

/** do propagation with Velocity-Verlet + Gaussian Thermostat
 * (update position)
 *
 * @param r	array for results position
 * @param dt	delta t
 * @param v	array for result velocity
 * @param F	array for Force
 * @param Fold	array for previous Force
 * @param minv	array for 1/mass
 * @param N number of particles (array size is 3N)
 * @param xi	\xi at (t)
 */
__global__ void propagateVelocityVerletGaussianThermo(real *r, real dt, real *v, real *F, real *Fold, real *minv, uint32_t N, real xi);

/** calculate the squared velocity (v^2) for each particles and store to tmp array
 *
 * @param v	array for velocity
 * @param tmp3N	array to store the results
 * @param N	number of particles
 */
__global__ void calcV2(real *v, real *tmp3N, uint32_t N);

/** calculate vx^2, vy^2, vz^2 for each particles and store to tmp array
 *
 * @param v	array for velocity
 * @param tmp3N	array to store the results
 * @param N	number of particles
 */
__global__ void calcV20(real *v, real *tmp3N, uint32_t N);

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
 * @param v	array for velocity
 * @param F	array to correct
 * @param m	array for mass of particles
 * @param lambda	correction term
 * @param N	number of particles
 */
__global__ void correctConstTemp(real *v, real *F, real *m, real lambda, uint32_t N);

/** calculate forces from SPH particles by Navier-Stokes equation
 *
 * @param r	array for position (size 3N)
 * @param a	results array for calculated accelerations (size 3N)
 * @param typeID	array for type ID of particles (size N)
 * @param N	number of particles
 * @param dW2D	2D dW/dr array for all i-j pair
 * @param rhoinv	reciprocal of mass density field (1/rho)
 * @param m	mass of particles
 * @param mu	array for shear viscosity
 * @param v	array for velocity field
 */
__global__ void calcF_SPH_NS(real *r, real *a, unsigned short *typeID, uint32_t N, real *dW2D, real *rhoinv, real *m, real *mu, real *v, const real K);

/** inspect density at SPH particles to treat particles on the free surface
 *
 * @param n	number density field
 * @param move	array for move flag
 * @param N	number of particles
 * @param R	result array for the density (only R[0] contains the result)
 */
__global__ void inspectDense(real *n, char *move, uint32_t N, real *R);

/** calculate ID of cutoff block
 * by block_Z + block_Y * num_Z + block_X * num_Z * num_Y
 *
 * calculated block ID and its particle ID is stored in bid[i] and pid[i] for i-th particle
 * 
 * @param r	position of particles
 * @param bid	block ID of particles
 * @param pid	particle ID 
 * @param N	number of particles
 * @param b0	length of block for X
 * @param b1	length of the block for Y
 * @param b2	length of the block for Z
 * @param cxmin	min of cell X
 * @param cymin	min of cell Y
 * @param czmin	min of cell Z
 * @param CX	number of the block in Z
 * @param CY	number of the block in Z*Y
 */
__global__ void calcBID(real *r, uint32_t *bid, uint32_t *pid, uint32_t N, real b0, real b1, real b2,
						real cxmin, real cymin, real czmin,
						uint32_t CX, uint32_t CY);

/** sort bid table in parallel, pid table is aloso modified by satisfing
 * particle pid[i] is located in bid[i]
 *
 * bindex table is also calculated, which is partial sum of number of particles
 * in each blocks
 *
 * @param bid	block ID of particles
 * @param pid	particle ID 
 * @param N	number of particles
 */
__global__ void sortByBID(uint32_t *pid, uint32_t *bid, uint32_t N);

/** makes/re-calculates bindex[] from sorted bid[]
 *
 * @param bid	block ID of particles
 * @param bindex	partial sum of number of particles
 * @param N	number of particles
 */
__global__ void makeBindex(uint32_t *bid, uint32_t *bindex, uint32_t N);

/** first phase to sort bid table by merge sort, bubble sort with a stride #thread x #MP
 *
 * pid table is aloso modified by satisfing
 * particle pid[i] is located in bid[i]
 *
 * bindex table is also calculated, which is partial sum of number of particles
 * in each blocks
 *
 * @param bid	block ID of particles
 * @param pid	particle ID 
 * @param N	number of particles
 */
__global__ void sortByBID_M1(uint32_t *pid, uint32_t *bid, uint32_t N);

/** acclumulate tmp81N array to the target array
 *
 * @param A	array for the target
 * @param A27	array with the size N (or 3N)*27
 * @param num	size of target array (N or 3N)
 * @param blocksize	size of A27/27, if (pitch==0) pitch = num
 * @param offset	offset in the array A27 for each num size block
 */
__global__ void reduce27(real *A, real *A27, uint32_t num, uint32_t pitch = 0, uint32_t offset = 0);

/** Restore an array by pid; dst[pid[i]] =src[i]
 *
 * @param dst	an array (size 3N)
 * @param src
 * @param N	array length
 * @param pid	particle ID map
 */
__global__ void RestoreByPid(real *dst, real *src, uint32_t N, uint32_t *pid);

/** merge Accelerations from other GPU
 * @param A		array for acceleration on my GPU
 * @param tmp3N	array for acceleration of other GPU
 * @param N		number of particles
 * @param myBlockOffset	offset of the Block ID on this GPU
 * @param myBlockNum	number of the blocks on this GPU
 */
__global__ void addArray(real *A, real *tmp3N, uint32_t N);

/** calculate the Eulerian Equation of Motion (Leap-Frog method)
 *
 * atitude and inertia are fixed, and initial inertia is 2/5 r^2 M E
 * 
 * \f[
 *  L_i += \Delta t * T_i \\
 *  \omega _i = \frac{1}{m_0} L_i
 * \f]
 */
__global__ void propagateEulerianRotation(real *w, real dt, real *L, real *T, real *m0inv, char *move, uint32_t N);

/** check cutoff blocks to select blocks which have movable particles
 *
 * @param selected	1D table for the block selection
 * @param pid		particle ID table (sorted by block ID)
 * @param move		move flags for each particles
 * @param bindex	index for the pid[] array for each blocks
 * @paran N		total number of the cutoff blocks
 */
__global__ void checkBlocks(uint32_t *selected, uint32_t *pid, char *move,
							uint32_t *bindex, uint32_t N);

/** accumulate 1D array with shared memory and reduction
 * (MPnum must be 1)
 *
 * @param selected	1D table for the block selection
 * @paran N		total number of the cutoff blocks
 * @param Res	results transfered from GPU's device memory to host memory
 */
__global__ void accumulate(uint32_t *selected, uint32_t N, real *Res);

/* write block ID to the selectedBlock[] table
 *
 * @param selected	1D table for the block selection
 * @paran N		total number of the cutoff blocks
 */
__global__ void writeBlockID(uint32_t *selected, uint32_t N);

/** cast real[] to uint32_t[] on GPU device memory
 *
 * @param r	array for real (src)
 * @param l array for uint32_t (dest)
 * @param N size of arrays
 * @param pid pid[] array to convert i=pid[_i] if given
 */
__global__ void real2ulong(real *r, uint32_t *l, uint32_t N, uint32_t *pid = NULL);

/** calculates partial sum with prefix scan
 *
 */
__global__ void partialsum(uint32_t *rowPtr, uint32_t N);

/** make list of particle j around i as colIdx[] by using Cutoff Block
 *
 * @param rowPrt row pointer for each i particle
 * @param colIdx	results array, colIdx[rowPtr[i]..rowPtr[i+1]] stores ID of particle j around i
 * @param r0	array for radius of each particles
 * @param r	array for position
 * @param blockOffset	offset for the block ID table
 * @param blockNeighbor	neighbor judgement for I-J block pair
 * @param pid	particle ID in sorted order
 * @param bindex	start particle num of this block
 * @param N	number of particles
 * @param selectedBlock	table for the selected blocks
 */
__global__ void makeJlist_WithBlock(uint32_t *rowPtr, uint32_t *colIdx,
									real *r0,
									real *r,
									uint32_t BlockOffset,
									uint32_t *blockNeighbor, uint32_t *pid, uint32_t *bindex,
									uint32_t N,
									uint32_t *selectedBlock = NULL);

/** make list of particle j around i as colIdx[] by using Cutoff Block
 *
 * @param rowPrt row pointer for each i particle
 * @param colIdx	results array, colIdx[rowPtr[i]..rowPtr[i+1]] stores ID of particle j around i
 * @param r0	array for radius of each particles
 * @param r	array for position
 * @param blockNeighbor	neighbor judgement for I-J block pair
 * @param pid	particle ID in sorted order
 * @param bindex	start particle num of this block
 * @param bid_by_pid	array for the Block ID can converted directly from pid
 * @param move  move/fixed particle table
 * @param N	number of particles
 */
__global__ void makeJlist_WithBlock2(const uint32_t *rowPtr, uint32_t *colIdx,
									 const real *r0,
									 const real *r,
									 const uint32_t *blockNeighbor, const uint32_t *pid, const uint32_t *bindex,
									 const uint32_t *bid_by_pid, const char *move,
									 const uint32_t N, const uint32_t Nstart = 0, const uint32_t Nend = 0);

__global__ void makeJlist_WithBlock4(const uint32_t *rowPtr, uint32_t *colIdx,
									 const real *r0,
									 const real *r,
									 const uint32_t *blockNeighbor, const uint32_t *bindex,
									 const uint32_t *bid, const char *move,
									 const uint32_t N, const uint32_t Nstart = 0, const uint32_t Nend = 0);

__global__ void calcBID_direct(const real *r, uint32_t *bid, uint32_t *bindex,
							   const uint32_t N, const real b0, const real b1, const real b2,
							   const real cxmin, const real cymin, const real czmin,
							   const uint32_t blen0, const uint32_t blen1, const uint32_t blen2,
							   uint32_t CX, uint32_t CY);

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
 * @param N	number of row
 */
__global__ void sortColIdx(const uint32_t *rowPtr, uint32_t *colIdx, uint32_t N);

__global__ void succeedPrevState(uint32_t *prevRow, uint32_t *prevCol, real *prevVal,
								 uint32_t *curRow, uint32_t *curCol, real *curVal,
								 uint32_t N);

#endif /* KERNELFUNCS */
