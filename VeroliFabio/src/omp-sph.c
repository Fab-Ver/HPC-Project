/**
 * Fabio Veroli 0000970669 fabio.veroli@studio.unibo.it
*/
/****************************************************************************
 *
 * sph.c -- Smoothed Particle Hydrodynamics
 *
 * https://github.com/cerrno/mueller-sph
 *
 * Copyright (C) 2016 Lucas V. Schuermann
 * Copyright (C) 2022 Moreno Marzolla
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* "Particle-Based Fluid Simulation for Interactive Applications" by
   Müller et al. solver parameters */

const float Gx = 0.0, Gy = -10.0;   // external (gravitational) forces
const float REST_DENS = 300;    // rest density
const float GAS_CONST = 2000;   // const for equation of state
const float H = 16;             // kernel radius
const float EPS = 16;           // equal to H
const float MASS = 2.5;         // assume all particles have the same mass
const float VISC = 200;         // viscosity constant
const float DT = 0.0007;        // integration timestep
const float BOUND_DAMPING = -0.5;

const int MAX_PARTICLES = 20000;
// Larger window size to accommodate more particles
#define WINDOW_WIDTH 3000
#define WINDOW_HEIGHT 2000

const int DAM_PARTICLES = 500;

const float VIEW_WIDTH = 1.5 * WINDOW_WIDTH;
const float VIEW_HEIGHT = 1.5 * WINDOW_HEIGHT;

/* Particle data structure; stores position, velocity, and force for
   integration stores density (rho) and pressure values for SPH.

   You may choose a different layout of the particles[] data structure
   to suit your needs. */
typedef struct {
    float x, y;         // position
    float vx, vy;       // velocity
    float fx, fy;       // force
    float rho, p;       // density, pressure
} particle_t;

particle_t *particles;
int n_particles = 0;    // number of currently active particles

/**
 * Return a random value in [a, b]
 */
float randab(float a, float b)
{
    return a + (b-a)*rand() / (float)(RAND_MAX);
}

/**
 * Set initial position of particle `*p` to (x, y); initialize all
 * other attributes to default values (zeros).
 */
void init_particle( particle_t *p, float x, float y )
{
    p->x = x;
    p->y = y;
    p->vx = p->vy = 0.0;
    p->fx = p->fy = 0.0;
    p->rho = 0.0;
    p->p = 0.0;
}

/**
 * Return nonzero iff (x, y) is within the frame
 */
int is_in_domain( float x, float y )
{
    return ((x < VIEW_WIDTH - EPS) &&
            (x > EPS) &&
            (y < VIEW_HEIGHT - EPS) &&
            (y > EPS));
}

/**
 * Initialize the SPH model with `n` particles. The caller is
 * responsible for allocating the `particles[]` array of size
 * `MAX_PARTICLES`.
 *
 * DO NOT parallelize this function, since it calls rand() which is
 * not thread-safe.
 *
 * For MPI and OpenMP: only the master must initialize the domain;
 *
 * For CUDA: the CPU must initialize the domain.
 */
void init_sph( int n )
{
    n_particles = 0;
    printf("Initializing with %d particles\n", n);

    for (float y = EPS; y < VIEW_HEIGHT - EPS; y += H) {
        for (float x = EPS; x <= VIEW_WIDTH * 0.8f; x += H) {
            if (n_particles < n) {
                float jitter = rand() / (float)RAND_MAX;
                init_particle(particles + n_particles, x+jitter, y);
                n_particles++;
            } else {
                return;
            }
        }
    }
    assert(n_particles == n);
}

/**
 * The original two loops of the serial version cannot be parallelized in an embarrassingly way because the inner loop can lead to race condition. 
 * In the provided solution, the loops are decomposed into three different loops: 
 *      1. The first loop initializes the rho properties of each particles. Embarrassingly parallel initialization can be partitioned across threads.
 *      2. The second loop combines the two original loops in a larger iteration, using the omp collapse clause. An omp atomic clause is necessary because 
 *         the rho properties can be concurrently accessed by different threads, potentially leading to race conditions when the (i) indexes of the iterations correspond.
 *      3. The third loop updates the p properties of each particles after the rho has been computed by the second loop (embarrassingly parallel).
 * The pool of threads is created at the beginning and then reused for every loop. This solution was preferred over parallelizing only the outer loop, 
 * because it has demonstrated better performance. 
*/
void compute_density_pressure( void )
{
    const float HSQ = H * H;    // radius^2 for optimization

    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float POLY6 = 4.0 / (M_PI * pow(H, 8));

#if __GNUC__ < 9
#pragma omp parallel default(none) shared(particles, n_particles)
#else
#pragma omp parallel default(none) shared(particles, n_particles, HSQ, POLY6, MASS, GAS_CONST, REST_DENS)
#endif
    {
        particle_t *pi = NULL;
#pragma omp for
        for(int i=0; i<n_particles; i++){
            pi = &particles[i];
            pi->rho = 0.0;
        }

#pragma omp for collapse(2)
        for (int i=0; i<n_particles; i++) {
            for (int j=0; j<n_particles; j++) {
                pi = &particles[i];
                const particle_t *pj = &particles[j];

                const float dx = pj->x - pi->x;
                const float dy = pj->y - pi->y;
                const float d2 = dx*dx + dy*dy;

                if (d2 < HSQ) {
#pragma omp atomic
                    pi->rho += MASS * POLY6 * pow(HSQ - d2, 3.0);
                }
            }
        }
        
#pragma omp for
        for(int i=0; i<n_particles; i++){
            pi = &particles[i];
            pi->p = GAS_CONST * (pi->rho - REST_DENS);
        }
    }
}

/**
 * The two loops inside the function contain dependencies that cross iteration boundaries, so they cannot be parallelized using the embarrassingly parallel pattern.
 * Specifically, each iteration of the outer loop is independent and can be parallelized by partitioning the iterations across threads.
 * However, the iterations of the inner loop are not independent as they update the common variables fpress_x, fpress_y, fvisc_x and fvisc_y, so can lead to race condition. 
 * In contrast to the previous function (compute_density_pressure()), only the outer loop is parallelized using the omp parallel for clause. 
 * This approach partitions the particles across threads, but each particle executes the calculations of the inner loop serially. 
 * This solution was preferred to the one of the previous function because the synchronization required to prevent race conditions introduces excessive overhead.  
*/
void compute_forces( void )
{
    /* Smoothing kernels defined in Muller and their gradients adapted
       to 2D per "SPH Based Shallow Water Simulation" by Solenthaler
       et al. */
    const float SPIKY_GRAD = -10.0 / (M_PI * pow(H, 5));
    const float VISC_LAP = 40.0 / (M_PI * pow(H, 5));
    const float EPS = 1e-6;
    
#if __GNUC__ < 9 
#pragma omp parallel for default(none) shared(particles, n_particles)
#else 
#pragma omp parallel for default(none) shared(particles, n_particles, SPIKY_GRAD, VISC_LAP, EPS, MASS, VISC, H, Gx, Gy)
#endif
    for (int i=0; i<n_particles; i++) {
        particle_t *pi = &particles[i];
        float fpress_x = 0.0, fpress_y = 0.0;
        float fvisc_x = 0.0, fvisc_y = 0.0;

        for (int j=0; j<n_particles; j++) {
            const particle_t *pj = &particles[j];

            if (pi == pj)
                continue;

            const float dx = pj->x - pi->x;
            const float dy = pj->y - pi->y;
            const float dist = hypotf(dx, dy) + EPS; // avoids division by zero later on

            if (dist < H) {
                const float norm_dx = dx / dist;
                const float norm_dy = dy / dist;
                // compute pressure force contribution
                fpress_x += -norm_dx * MASS * (pi->p + pj->p) / (2 * pj->rho) * SPIKY_GRAD * pow(H - dist, 3);
                fpress_y += -norm_dy * MASS * (pi->p + pj->p) / (2 * pj->rho) * SPIKY_GRAD * pow(H - dist, 3);
                // compute viscosity force contribution
                fvisc_x += VISC * MASS * (pj->vx - pi->vx) / pj->rho * VISC_LAP * (H - dist);
                fvisc_y += VISC * MASS * (pj->vy - pi->vy) / pj->rho * VISC_LAP * (H - dist);
            }
        }
        const float fgrav_x = Gx * MASS / pi->rho;
        const float fgrav_y = Gy * MASS / pi->rho;
        pi->fx = fpress_x + fvisc_x + fgrav_x;
        pi->fy = fpress_y + fvisc_y + fgrav_y;
    }
}

/**
 * Since there are no dependencies crossing iteration boundaries, each iteration of the loop
 * can be executed by any thread in any order, allowing for full parallelization (embarrassingly parallel).
*/
void integrate( void )
{
#if __GNUC__ < 9
#pragma omp parallel for default(none) shared(particles, n_particles)
#else
#pragma omp parallel for default(none) shared(particles, n_particles, DT, EPS, BOUND_DAMPING, VIEW_WIDTH, VIEW_HEIGHT)
#endif
    for (int i=0; i<n_particles; i++) {
        particle_t *p = &particles[i];
        // forward Euler integration
        p->vx += DT * p->fx / p->rho;
        p->vy += DT * p->fy / p->rho;
        p->x += DT * p->vx;
        p->y += DT * p->vy;

        // enforce boundary conditions
        if (p->x - EPS < 0.0) {
            p->vx *= BOUND_DAMPING;
            p->x = EPS;
        }
        if (p->x + EPS > VIEW_WIDTH) {
            p->vx *= BOUND_DAMPING;
            p->x = VIEW_WIDTH - EPS;
        }
        if (p->y - EPS < 0.0) {
            p->vy *= BOUND_DAMPING;
            p->y = EPS;
        }
        if (p->y + EPS > VIEW_HEIGHT) {
            p->vy *= BOUND_DAMPING;
            p->y = VIEW_HEIGHT - EPS;
        }
    }
}

/**
 * The function can be easily parallelized using the reduction pattern, 
 * particularly by utilizing the OpenMP reduction clause. 
*/
float avg_velocities( void )
{
    double result = 0.0;
#pragma omp parallel for reduction(+:result) default(none) shared(particles, n_particles)
    for (int i=0; i<n_particles; i++) {
        result += hypot(particles[i].vx, particles[i].vy) / n_particles;
    }
    return result;
}

void update( void )
{
    compute_density_pressure();
    compute_forces();
    integrate();
}

int main(int argc, char **argv)
{
    srand(1234);

    particles = (particle_t*)malloc(MAX_PARTICLES * sizeof(*particles));
    assert( particles != NULL );

    int n = DAM_PARTICLES;
    int nsteps = 50;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [nparticles [nsteps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    if (n > MAX_PARTICLES) {
        fprintf(stderr, "FATAL: the maximum number of particles is %d\n", MAX_PARTICLES);
        return EXIT_FAILURE;
    }

    if (n <= 0){
        fprintf(stderr, "ERROR: enter a positive number of particles\n");
        return EXIT_FAILURE;
    }

    if (nsteps <= 0){
        fprintf(stderr, "ERROR: enter a positive number of steps\n");
        return EXIT_FAILURE;
    }

    init_sph(n);

    double time_start = hpc_gettime();
    for (int s=0; s<nsteps; s++) {
        update();
        /* the average velocities MUST be computed at each step, even
           if it is not shown (to ensure constant workload per
           iteration) */
        const float avg = avg_velocities();
        if (s % 10 == 0)
            printf("step %5d, avgV=%f\n", s, avg);
    }
    double time_elapsed = hpc_gettime() - time_start;
    printf("Elapsed Time: %f\n", time_elapsed);
    
    free(particles);
    return EXIT_SUCCESS;
}
