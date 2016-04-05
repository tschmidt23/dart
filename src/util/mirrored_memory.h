#ifndef MIRRORED_MEMORY_H
#define MIRRORED_MEMORY_H

#include <cstddef>
#include <cuda_runtime.h>
#include "geometry/grid_3d.h"
#include "util/cuda_utils.h"

namespace dart {

// -=-=-=-=-=-=-=-=-=- MirroredData interface -=-=-=-=-=-=-=-=-=-

/**
 * @brief MirroredData is a super class for various data types that are mirrored between device and host.
 */
class MirroredData {
public:

    /**
     * This function copies the device-resident memory to the host.
     */
    virtual void syncHostToDevice() = 0;

    /**
     * This function copies the host-resident memory to the device.
     */
    virtual void syncDeviceToHost() = 0;
};

// -=-=-=-=-=-=-=-=-=- Vector -=-=-=-=-=-=-=-=-=-

template <typename T>
/**
 * @brief MirroredVector is similar to std::vector, but mirrored on the device.
 */
class MirroredVector : public MirroredData {
public:

    /**
     * This will create a MirroredVector of the specified length, allocating the required amount of memory.
     * @param length The number of elements in the vector.
     */
    MirroredVector(const uint length);

    MirroredVector(const MirroredVector & other);

    ~MirroredVector();
    inline void syncHostToDevice() { cudaMemcpy(_dVector,_hVector,_length*sizeof(T),cudaMemcpyHostToDevice); }
    inline void syncDeviceToHost() { cudaMemcpy(_hVector,_dVector,_length*sizeof(T),cudaMemcpyDeviceToHost); }

    /**
     * This returns a pointer to the host-resident data.
     * @return The host pointer.
     */
    inline T * hostPtr() { return _hVector; }
    inline const T * hostPtr() const { return _hVector; }

    /**
     * This returns a pointer to the device-resident data.
     * @return The device pointer.
     */
    inline T * devicePtr() { return _dVector; }
    inline const T * devicePtr() const { return _dVector; }

    /**
     * This returns the number of elements in the vector.
     * @return The length of the vector.
     */
    inline uint length() const { return _length; }

    void resize(const uint length);

    T & operator[](const int i) { return _hVector[i]; }

    const T & operator[](const int i) const { return _hVector[i]; }

    MirroredVector<T> & operator= (const MirroredVector<T> & other);

private:
    T * _hVector;
    T * _dVector;
    uint _length;
};

template <typename T>
MirroredVector<T>::MirroredVector(const uint length) : _length(length) {
    if(length != 0){
        cudaMallocHost(&_hVector,length*sizeof(T));
        cudaMalloc(&_dVector,length*sizeof(T));
    }
    else{
        _hVector = NULL;
        _dVector = NULL;
    }
}

template <typename T>
MirroredVector<T>::MirroredVector(const MirroredVector & other) {
    _length = other.length();
    if(_length){
        cudaMallocHost(&_hVector,_length*sizeof(T));
        cudaMalloc(&_dVector,_length*sizeof(T));
        cudaMemcpy(_hVector,other._hVector,_length*sizeof(T),cudaMemcpyHostToHost);
        cudaMemcpy(_dVector,other._dVector,_length*sizeof(T),cudaMemcpyDeviceToDevice);
    }
    else{
        _hVector = NULL;
        _dVector = NULL;
    }
}


template <typename T>
MirroredVector<T>::~MirroredVector() {
    cudaFreeHost(_hVector);
    cudaFree(_dVector);
}

template <typename T>
void MirroredVector<T>::resize(const uint length) {
    T * hTmp = _hVector;
    T * dTmp = _dVector;
    if(length != 0){
        // host
        cudaMallocHost(&_hVector,length*sizeof(T));
        cudaMemcpy(_hVector,hTmp,std::min(_length,length)*sizeof(T),cudaMemcpyHostToHost);
        
        // device
        cudaMalloc(&_dVector,length*sizeof(T));
        cudaMemcpy(_dVector,dTmp,std::min(_length,length)*sizeof(T),cudaMemcpyDeviceToDevice);
    }
    else{
        _hVector = NULL;
        _dVector = NULL;
    }
    cudaFreeHost(hTmp);
    cudaFree(dTmp);
    _length = length;
}

template <typename T>
MirroredVector<T> & MirroredVector<T>::operator= (const MirroredVector<T> & other) {
    resize(other._length);
    if(_length != 0){
        memcpy(_hVector,other._hVector,_length*sizeof(T));
        cudaMemcpy(_dVector,other._dVector,_length*sizeof(T),cudaMemcpyDeviceToDevice);
    }
    return *this;
}

// -=-=-=-=-=-=-=-=-=- Grid3D -=-=-=-=-=-=-=-=-=-
template <typename T>
/**
 * @brief MirroredGrid3D class provides a mirrored implementation of the Grid3D class.
 */
class MirroredGrid3D : public MirroredData {
public:

    /**
     * @brief A MirroredGrid3D instance must be constructed from a stricly host-resident Grid3D instance of the same templated type. The constructor allocates all appropriate memory and copies the data from the given Grid3D object to both device and host.
     * @param grid The Grid3D object to copy to the mirrored object.
     */
    MirroredGrid3D<T>(const Grid3D<T> & grid);
    ~MirroredGrid3D();
    inline void syncHostToDevice() {
        cudaMemcpy(_dData,_hData,_dataSize,cudaMemcpyHostToDevice);
        _hGrid->data = _dData;
        cudaMemcpy(_dGrid,_hGrid,sizeof(Grid3D<T>),cudaMemcpyHostToDevice);
        _hGrid->data = _hData;
    }
    inline void syncDeviceToHost() {
        cudaMemcpy(_hData,_dData,_dataSize,cudaMemcpyDeviceToHost);
        cudaMemcpy(_hGrid,_dGrid,sizeof(Grid3D<T>),cudaMemcpyDeviceToHost);
        _hGrid->data = _hData;
    }

    /**
     * This returns a pointer to the copy of the grid on the host.
     * @return A pointer to the host-resident grid.
     */
    inline Grid3D<T> * hostGrid() { return _hGrid; }

    /**
     * This returns a pointer to the copy of the grid on the device.
     * @return A pointer to the device-resident grid.
     */
    inline Grid3D<T> * deviceGrid() { return _dGrid; }

    /**
     * This returns a pointer directly to the linearized data of the host-resident grid. It is equivalent to hostGrid()->data.
     * @return A pointer to the host-resident grid data.
     */
    inline T * hostData() { return _hData; }

    /**
     * This returns a pointer directly to the linearized data of the device-resident grid. It is equivalent to deviceGrid()->data.
     * @return A pointer to the device-resident grid data.
     */
    inline T * deviceData() { return _dData; }

private:
    Grid3D<T> * _hGrid;
    Grid3D<T> * _dGrid;
    T * _hData;
    T * _dData;
    const size_t _dataSize;
};

template <typename T>
MirroredGrid3D<T>::MirroredGrid3D(const Grid3D<T> & grid) : _dataSize(grid.dim.x*grid.dim.y*grid.dim.z*sizeof(T)) {
    cudaMallocHost(&_hGrid,sizeof(Grid3D<T>));
    cudaMalloc(&_dGrid,sizeof(Grid3D<T>));
    cudaMallocHost(&_hData,_dataSize);
    cudaMalloc(&_dData,_dataSize);
    memcpy(_hData,grid.data,_dataSize);
    memcpy(_hGrid,&grid,sizeof(Grid3D<T>));
    _hGrid->data = _hData;
    syncHostToDevice();
}

template <typename T>
MirroredGrid3D<T>::~MirroredGrid3D() {
    cudaFreeHost(_hGrid);
    cudaFree(_dGrid);
    cudaFreeHost(_hData);
    cudaFree(_dData);
}

// -=-=-=-=-=-=-=-=-=- Grid3DVector -=-=-=-=-=-=-=-=-=-
template <typename T>
/**
 * @brief The MirroredGrid3DVector class provides a deep-mirrored vector of Grid3D instances. That is to say that the instances are mirrored, as well as the vector of pointers to the grids.
 */
class MirroredGrid3DVector : public MirroredData {
public:

    /**
     * A MirroredGrid3DVector instance must be constructed from an array of stricly host-resident Grid3D instances of the same templated type. The constructor allocates all appropriate memory and copies the data from the given Grid3D objects to both device and host.
     * @param length The number of grids to be mirrored.
     * @param grids An array of pointers to Grid3D instances of size length. The constructor will allocate all appropriate memory and copy the data from all grids to both device and host. The MirroredGrid3DVector instance does not own these pointers.
     */
    MirroredGrid3DVector(const uint length, const Grid3D<T> * grids);
    ~MirroredGrid3DVector();
    inline void syncHostToDevice() {
        for (int i=0; i<_length; ++i) {
            _hGrids[i].data = _dDatas[i];
            cudaMemcpy(_dDatas[i],_hDatas[i],_dataSizes[i],cudaMemcpyHostToDevice);
        }
        cudaMemcpy(_dGrids,_hGrids,_length*sizeof(Grid3D<T>),cudaMemcpyHostToDevice);
        for (int i=0; i<_length; ++i) {
            _hGrids[i].data = _hDatas[i];
        }
    }
    inline void syncDeviceToHost() {
        cudaMemcpy(_hGrids,_dGrids,_length*sizeof(Grid3D<T>),cudaMemcpyDeviceToHost);
        for (int i=0; i<_length; ++i) {
            cudaMemcpy(_hDatas[i],_dDatas[i],_dataSizes[i],cudaMemcpyDeviceToHost);
            _hGrids[i].data = _hDatas[i];
        }
    }

    /**
     * This returns a pointer to the first host-resident Grid3D instance. A pointer to (host-resident) instance i (for i < length) can be accessed via hostGrids()[i].
     * @return The pointer to the first host-resident Grid3D instance.
     */
    inline Grid3D<T> * hostGrids() { return _hGrids; }

    /**
     * This returns a pointer to the first device-resident Grid3D instance. A pointer to (device-resident) instance i (for i < length) can be accessed via deviceGrids()[i].
     * @return The pointer to the first device-resident Grid3D instance.
     */
    inline Grid3D<T> * deviceGrids() { return _dGrids; }
private:
    Grid3D<T> * _hGrids;
    Grid3D<T> * _dGrids;
    T * * _hDatas;
    T * * _dDatas;
    size_t * _dataSizes;
    const uint _length;
};

template <typename T>
MirroredGrid3DVector<T>::MirroredGrid3DVector(const uint length, const Grid3D<T> * grids) : _length(length) {
    cudaMallocHost(&_hGrids,length*sizeof(Grid3D<T>));
    cudaMalloc(&_dGrids,length*sizeof(Grid3D<T>));
    _dataSizes = new size_t[length];
    _hDatas = new T*[length];
    _dDatas = new T*[length];
    memcpy(_hGrids,grids,length*sizeof(Grid3D<T>));
    for (int i=0; i<length; ++i) {
        const Grid3D<T> &vg = grids[i];
        _dataSizes[i] = vg.dim.x*vg.dim.y*vg.dim.z*sizeof(T);
        cudaMallocHost(&_hDatas[i],_dataSizes[i]);
        cudaMalloc(&_dDatas[i],_dataSizes[i]);
        memcpy(_hDatas[i],grids[i].data,_dataSizes[i]);
        _hGrids[i].data = _hDatas[i];
    }

}

template <typename T>
MirroredGrid3DVector<T>::~MirroredGrid3DVector() {
    cudaFreeHost(_hGrids);
    cudaFree(_dGrids);
    for (int i=0; i<_length; ++i) {
        cudaFreeHost(_hDatas[i]);
        cudaFree(_dDatas[i]);
    }
    delete [] _dataSizes;
    delete [] _hDatas;
    delete [] _dDatas;
}

}

#endif // MIRRORED_MEMORY_H
