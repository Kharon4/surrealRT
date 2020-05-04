#define _INSIDE_commonMemory_Header 1
#include "cudaRelated/commonMemory.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
commonMemory<T>::commonMemory(size_t Size, commonMemType Type) {
	size = sizeof(T) * Size;
	type = Type;
	if (size == 0)return;

	if (type != commonMemType::deviceOnly)
		hostPtr = new unsigned char[size];

	if (type != commonMemType::hostOnly)
		cudaMalloc(&devicePtr, size);
}

template <typename T>
T* commonMemory<T>::getHost() {
	if (type == deviceOnly)return (T*)nullptr;
	if (!hostUpdated && type == both) {
		cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		hostUpdated = true;
	}
	return (T*)hostPtr;
}

template <typename T>
T* commonMemory<T>::getDevice() {
	if (type == hostOnly)return (T*)nullptr;
	if (hostUpdated && type == both) {
		cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
		hostUpdated = false;
	}
	return (T*)devicePtr;
}

template <typename T>
commonMemory<T>::~commonMemory() {
	if (hostPtr != nullptr)delete[] hostPtr;
	if (devicePtr != nullptr)cudaFree(devicePtr);

}