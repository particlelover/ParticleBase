#if !defined(KERNELINLINE)
#define KERNELINLINE
#include <assert.h>
#include "kernelfuncs.h"

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)) || defined(__APPLE_CC__)
#undef assert
#define assert(arg)
#endif

/*
 * common inline functions
 */
/** calculate distance in between two points and returns squared of this distance
 *
 *  @return	(distance r)^2
 */
inline __device__ real distance2(real r1x, real r1y, real r1z,
																 const real r2x, const real r2y, const real r2z)
{
	r1x -= r2x;
	r1y -= r2y;
	r1z -= r2z;
	r1x *= r1x;
	r1y *= r1y;
	r1z *= r1z;

	return r1x + r1y + r1z;
}
/** calculate distance in between two points and returns squared of this distance
 * with periodic boundary condition treatment
 *
 *  @param cx	(cell_x / 2)^2; if |r1x-r2x|>cx/2; then (cell-|r1x-r2x|) is calculated
 *  @return	(distance r)^2
 */
inline __device__ real distance2p(real r1x, real r1y, real r1z,
																	const real r2x, const real r2y, const real r2z,
																	const real cx, const real cy, const real cz)
{
	// cx should be (cell_x/2)^2
	r1x -= r2x;
	r1y -= r2y;
	r1z -= r2z;
	r1x *= r1x;
	r1y *= r1y;
	r1z *= r1z;
	if (r1x > cx)
		r1x += 4 * (cx - sqrt(cx * r1x));
	if (r1y > cy)
		r1y += 4 * (cy - sqrt(cy * r1y));
	if (r1z > cz)
		r1z += 4 * (cz - sqrt(cz * r1z));

	return r1x + r1y + r1z;
}

/*
 * particleMD
 */
#define MAXELEMNUM 5
//! table for LJ parameter sigma and epsilon for all i-j types pair at constant memory
//! as CUDA's float2 or double2 for \f$ \sigma^2 \f$ and \f$ 24\epsilon \f$
__constant__ REAL2 cLJparams[MAXELEMNUM * MAXELEMNUM];

/** function object to calculate the force from LJ potential
 *
 * operator() should be inline because it is also __device__ function
 */
class potentialLJ
{
public:
	/** calculate force from LJ potential
 *
 * @param r2	squared distance (r^2)
 * @param i	type ID of particle i
 * @param j	type ID of particle j
 * @return	force
 */
	__device__ real operator()(const real r2, const unsigned short i, const unsigned short j) const
	{
		const REAL2 para = cLJparams[i * MAXELEMNUM + j];
		const real s2 = para.x;
		const real e24 = para.y;
		real d1 = s2 / r2;				 // sigma^2/r^2
		d1 *= d1 * d1;						 // sigma^6/r^6
															 //	const real d2 = d1*d1;	// sigma^12/r^12
		const real _r = rsqrt(r2); // 1/r
															 //	return e24*(2*d2*_r - d1*_r);
		return e24 * d1 * (2 * d1 - 1) * _r;
	}
};

/** set LJ parameter sigma and epsilon for all i-j types pair into GPU's constant memory
 *
 * both setLJparams and optentialLJ should be defined same file 
 * to share the cLJparams[]
 *
 * @param p	vetor for LJ parameters as sigma, epsilon for all i-j types pair
 * @param elemnum	number of elements (<=MAXELEMNM)
 */
inline __host__ void _setLJparams(const std::vector<real> &p, uint32_t elemnum)
{
	assert(elemnum <= MAXELEMNUM);

	const REAL2 p0 = {0.0, 0.0};
	std::vector<REAL2> p2(MAXELEMNUM * MAXELEMNUM, p0);
	for (int i = 0; i < elemnum; ++i)
		for (int j = 0; j < elemnum; ++j)
		{
			REAL2 _p;
			_p.x = p[(i * elemnum + j) * 2] * p[(i * elemnum + j) * 2]; // \sigma ^2
			_p.y = p[(i * elemnum + j) * 2 + 1] * 24;										// 24 \epsilon
			p2[i * MAXELEMNUM + j] = _p;
		}

	cudaMemcpyToSymbol(cLJparams, &(p2[0]), sizeof(REAL2) * MAXELEMNUM * MAXELEMNUM);
}

/** function object to calculate the force from softcore potential
 *
 * operator() should be inline because it is also __device__ function
 */
class potentialSC
{
public:
	real rc; //!< r cutoff; f(r/rc*PI/2)=0
	real f0; //!< strength of the force

	/** calculate force from SoftCore potential
   *
   * @param r2	squared distance (r^2)
   * @param i	type ID of particle i
   * @param j	type ID of particle j
   * @return	force
   */
	__device__ real operator()(const real r2, const unsigned short i, const unsigned short j) const
	{
		const real r = sqrt(r2);
		real res = f0 * cos(r / rc * M_PI / 2);
		if (r > rc)
			res = 0;
		return res;
	}
};

/*
 * particleSPH
 */
/** Lucy kernel 3D
 *
 * \f[
 * \omega (r, h) = \frac{105}{16\pi h^3}(1+3\frac{r}{h})(1-\frac{r}{h})^3
 * = \frac{105}{16\pi h^7}(h+3r)(h-r)^3
 * \f]
 *
 * w0 = 105/(16pi h^7)
 */
class SPHKernelLucy
{
public:
	/** calculate the SPH kernel by Lucy 3D
 *
 * @param r	array for position (size 3N)
 * @param h	SPH kernel radius
 * @param w0	coefficient for SPH kernel
 * @return	kernel's value
 */
	__device__ real operator()(real r, real h, real w0) const
	{

		real _h = h - r;
		if (0 > _h)
			return 0;
		_h *= _h * _h;
		return w0 * (h + 3 * r) * _h;
	}
};

/** derivative of Lucy kernel (dw/dr / r)
 * 
 * \f[
 * \frac{\partial \omega (r, h)}{\partial r} \frac{1}{r} = 
 * = -12\frac{105}{16\pi h^7}(h-r)^2
 * \f]
 *
 * w1 = -12 * 105/(16pi h^7) = -12*w0
 */
class SPHKernelLucyDW
{
public:
	/** calculate the gradient of SPH kernel by Lucy 3D
 *
 * @param r	array for position (size 3N)
 * @param h	SPH kernel radius
 * @param w1	coefficient for SPH kernel gradient
 * @return	kernel's value
 */
	__device__ real operator()(real r, real h, real w1) const
	{

		real _h = h - r;
		if (0 > _h)
			return 0;
		return w1 * _h * _h;
	}
};

/*
 * cudaParticleDEM
 */
/** calculate contact forces/torques for the DEM
 *
 * @param r	position
 * @param F force (results)
 * @param v	velocity
 * @param N	number of particles
 * @param w	angular velocity
 * @param T	Torque (result)
 * @param r0	radius of DEM particle
 * @param E		=\f$Y/(1-\sigma ^2)\f$ from Young Modulus/Poisson ratio
 * @param mu	friction constant
 * @param s2	=\f$[(2-\sigma )/2/(1-\sigma )]\f$
 * @param gamma2	2*gamma; \f$ \gamma = - \frac{\ln e}{\sqrt{\pi ^2 + (\ln e)^2}} \f$; \f$ e\f$
 * @param mu_r	friction for rotation
 */
class DEMContact
{
public:
	real *v, *w;
	real *r0;
	char *move;
	real E, mu, s2, gamma2, mu_r;
	real *minv;

	uint32_t *rowPtr;
	uint32_t *colIdx;
	real *val;
	real deltaT;
	__device__ void operator()(const real *_r, const real *r, uint32_t N, uint32_t N2, uint32_t i, uint32_t j, real *_f0) const
	{
		if (i != j)
		{
			const real R2 = distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]);
			const real rc2 = r0[i] + r0[j];
			if (R2 < rc2 * rc2)
			{
				// contact
				real n[3];
				n[0] = _r[0] - r[j];
				n[1] = _r[1] - r[j + N];
				n[2] = _r[2] - r[j + N2];

				real Rij[3];
				Rij[0] = -r0[i] / (r0[i] + r0[j]) * n[0];
				Rij[1] = -r0[i] / (r0[i] + r0[j]) * n[1];
				Rij[2] = -r0[i] / (r0[i] + r0[j]) * n[2];

				// normal vector
				n[0] *= rsqrt(R2);
				n[1] *= rsqrt(R2);
				n[2] *= rsqrt(R2);

				real vij[3];
				vij[0] = v[i] - v[j];
				vij[1] = v[i + N] - v[j + N];
				vij[2] = v[i + N2] - v[j + N2];

				// Force for normal
				//real xi_n = std::max(r0[i]+r0[j]-sqrt(R2), 0.0);
				real xi_n = r0[i] + r0[j] - sqrt(R2);
				// contact case: r0_i + r0_j > sqrt(R2)
				// therefore xi_n should larger than 0
				//if (xi_n<0) xi_n = 0;
				// if xi_n==0, xi_s_max = mu*s2*xi_n becomes 0, and divide by xi_s_max will lead to be diverged
				assert(xi_n > 0.0);
				const real C3 = vij[0] * n[0] + vij[1] * n[1] + vij[2] * n[2];
				real Fn[3] = {0.0, 0.0, 0.0};
				{
					const real Rbar = (r0[i] * r0[j]) / (r0[i] + r0[j]);
					const real mbar = 1.0 / (minv[i] + minv[j]);
					const real Kn = 2.0 / 3.0 * E * sqrt(Rbar * xi_n);
					const real C4 = Kn * xi_n - gamma2 * sqrt(mbar * Kn) * C3;
					Fn[0] = n[0] * C4;
					Fn[1] = n[1] * C4;
					Fn[2] = n[2] * C4;
					// Cundall and Strack (1979) condition
					const real dt_crit = 2 * sqrt(mbar / Kn);
					assert(deltaT < dt_crit);
				}

				// Force for tangential
				real Fs[3] = {0.0, 0.0, 0.0};
				const real F0 = sqrt(Fn[0] * Fn[0] + Fn[1] * Fn[1] + Fn[2] * Fn[2]);
				{
					real vijt[3];
					vijt[0] = vij[0] - C3 * n[0];
					vijt[1] = vij[1] - C3 * n[1];
					vijt[2] = vij[2] - C3 * n[2];
					vijt[0] += (w[i + N] + w[j + N]) * Rij[2] - (w[i + N2] + w[j + N2]) * Rij[1];
					vijt[1] += (w[i + N2] + w[j + N2]) * Rij[0] - (w[i] + w[j]) * Rij[2];
					vijt[2] += (w[i] + w[j]) * Rij[1] - (w[i + N] + w[j + N]) * Rij[0];
					if ((vijt[0] != 0) && (vijt[1] != 0) && (vijt[2] != 0))
					{
						const real xi_s_max = mu * s2 * xi_n;

						const real v0 = rsqrt(vijt[0] * vijt[0] + vijt[1] * vijt[1] + vijt[2] * vijt[2]);
						//const real v1 = - mu * F0 * std::min(xi_s[i], xi_s_max) / xi_s_max * v0;

						real xi_s_ij = 0;
						uint32_t _q = 0;
						for (uint32_t _p = rowPtr[i]; _p < rowPtr[i + 1]; ++_p)
						{
							if (j == colIdx[_p])
								_q = _p;
						}
						xi_s_ij = val[_q];
						const real v1 = -mu * F0 * ((xi_s_ij < xi_s_max) ? (xi_s_ij / xi_s_max) : 1.0) * v0;
						//assert(v0==v0);
						//assert(v1==v1);
						Fs[0] = v1 * vijt[0];
						Fs[1] = v1 * vijt[1];
						Fs[2] = v1 * vijt[2];

						const real __V = vijt[0] * vijt[0] + vijt[1] * vijt[1] + vijt[2] * vijt[2];
						val[_q] += sqrt(__V) * deltaT;
					}
				}

				// Torque
				real Ts[3];
				Ts[0] = Rij[1] * Fs[2] - Rij[2] * Fs[1];
				Ts[1] = Rij[2] * Fs[0] - Rij[0] * Fs[2];
				Ts[2] = Rij[0] * Fs[1] - Rij[1] * Fs[0];

				// Friction (rotation)
				real Tr[3] = {0.0, 0.0, 0.0};
				if ((w[i] != 0) && (w[i + N] != 0) && (w[i + N2] != 0))
				{
					const real w0 = rsqrt(w[i] * w[i] + w[i + N] * w[i + N] + w[i + N2] * w[i + N2]);
					Tr[0] = -mu_r * r0[i] * F0 * w0 * w[i];
					Tr[1] = -mu_r * r0[i] * F0 * w0 * w[i + N];
					Tr[2] = -mu_r * r0[i] * F0 * w0 * w[i + N2];
				}

				// update force and torque
				const int Q = 2 - move[j]; // 2 for fixed wall, otherwise 1
				_f0[0] += (Fn[0] + Fs[0]) * Q;
				_f0[1] += (Fn[1] + Fs[1]) * Q;
				_f0[2] += (Fn[2] + Fs[2]) * Q;

				real *_t0 = &(_f0[3]);
				_t0[0] += (Ts[0] + Tr[0]) * Q;
				_t0[1] += (Ts[1] + Tr[1]) * Q;
				_t0[2] += (Ts[2] + Tr[2]) * Q;
			}
		}
	}
};

/*
 * cudaSparseMat
 */
/** calculates coordination number around i particle
 *
 */
class calcCoord
{
public:
	real *r0; //!< 1D array for the radius of particles

	__device__ void operator()(const real *_r, const real *r, uint32_t N, uint32_t N2, uint32_t i, uint32_t j, real *_f0) const
	{
		if (i != j)
		{
			const real R2 = distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]);
			const real rc2 = r0[i] + r0[j];
			if (R2 < rc2 * rc2)
			{
				_f0[0] += 1.0;
			}
		}
	}
};
#endif
