#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>

#define CheckCudaDieOnError() _CheckCudaDieOnError( __FILE__, __LINE__ );
namespace dart {
inline void _CheckCudaDieOnError( const char * sFile, const int nLine ) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string errorString(cudaGetErrorString(error));
        std::cerr << "CUDA error: " << errorString << std::endl;
        std::cerr << "from line " << nLine << " of file " << sFile << std::endl;
        exit(1);
    }
}
} // namespace dart


#endif // CUDA_UTILS_H
