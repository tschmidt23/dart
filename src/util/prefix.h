#ifndef PREFIX_H
#define PREFIX_H

#include <host_defines.h>

#ifdef CUDA_BUILD
#define PREFIX __device__ __host__
#else
#define PREFIX
#endif // CUDA_BUILD

#endif // PREFIX_H
