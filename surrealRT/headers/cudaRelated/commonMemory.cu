#define _INSIDE_commonMemory_Header 1
#include "cudaRelated/commonMemory.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define commonMemDebugging

#ifdef  commonMemDebugging
//debugging
#include <iostream>
#endif //  commonMemDebugging



template <typename T>
commonMemory<T>::commonMemory(size_t Size, commonMemType Type) {
	noElements = Size;
	size = sizeof(T) * Size;
	type = Type;
	if (size == 0)return;

	if (type != commonMemType::deviceOnly)
		hostPtr = new unsigned char[size];

	if (type != commonMemType::hostOnly)
		cudaMalloc(&devicePtr, size);
}

template <typename T>
size_t commonMemory<T>::getNoElements() {
	return noElements;
}

template <typename T>
void commonMemory<T>::changeNoElements(size_t Size) {
	if (hostPtr != nullptr)delete[] hostPtr;
	if (devicePtr != nullptr)cudaFree(devicePtr);

	noElements = Size;
	size = sizeof(T) * Size;
	
	if (size == 0)return;

	if (type != commonMemType::deviceOnly)
		hostPtr = new unsigned char[size];

	if (type != commonMemType::hostOnly)
		cudaMalloc(&devicePtr, size);
}

template <typename T>
T* commonMemory<T>::getHost(bool* OUTupdated) {
	if (OUTupdated != nullptr)*OUTupdated = false;

	if (type == deviceOnly)return (T*)nullptr;
	if (!hostUpdated && type == both) {
		cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		hostUpdated = true;
		if (OUTupdated != nullptr)*OUTupdated = true;
	}
	return (T*)hostPtr;
}

template <typename T>
T* commonMemory<T>::getDevice(bool* OUTupdated) {
	if (OUTupdated != nullptr)*OUTupdated = false;

	if (type == hostOnly)return (T*)nullptr;
	if (hostUpdated && type == both) {
		cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
		hostUpdated = false;
		if (OUTupdated != nullptr)*OUTupdated = true;
	}
	return (T*)devicePtr;
}


template <typename T>
void commonMemory<T>::changeMemType(commonMemType newType) {
	//check if new type and old type r same.
	if (newType == type)return;
	
	//if !both create space
	if (type != commonMemType::both) {
		if (type == commonMemType::hostOnly) {
			//create device mem
			cudaMalloc(&devicePtr, size);
			//copy data
			cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
		}
		else {
			//create host mem
			hostPtr = new unsigned char[size];
			//copy data
			cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		}
	}
	else {
		//update the correct side;
		getHost();
		getDevice();
	}

	//delete
	if (newType == commonMemType::hostOnly) {
		hostUpdated = true;
		//delete device
		if (devicePtr != nullptr)cudaFree(devicePtr);
		devicePtr = nullptr;
	}
	else if (newType == commonMemType::deviceOnly) {
		hostUpdated = false;
		//delete host
		if (hostPtr != nullptr)delete[] hostPtr;
		hostPtr = nullptr;
	}

	//perform conversion
	type = newType;
}

template <typename T>
commonMemType commonMemory<T>::getMemType() {
	return type;
}




template <typename T>
void commonMemory<T>::shallowCopy(commonMemory<T>& other) {

#ifdef  commonMemDebugging
	std::cout << "shallow copy made\n";
#endif
	//delete stuff
	if (hostPtr != nullptr)delete[] hostPtr;
	if (devicePtr != nullptr)cudaFree(devicePtr);

	//copy stuff
	hostPtr = other.hostPtr;
	devicePtr = other.devicePtr;
	hostUpdated = other.hostUpdated;
	noElements = other.noElements;
	size = other.size; 
	type = other.type;

	//dissable other
	other.hostPtr = nullptr;
	other.devicePtr = nullptr;

}

template <typename T>
void commonMemory<T>::deepCopy(const commonMemory<T>& other) {

#ifdef  commonMemDebugging
	std::cout << "deep copy made\n";
#endif
	//delete stuff
	if (hostPtr != nullptr)delete[] hostPtr;
	if (devicePtr != nullptr)cudaFree(devicePtr);

	//update stuff
	hostUpdated = other.hostUpdated;
	noElements = other.noElements;
	size = other.size;
	type = other.type;

	//allocate and copy mem
	if (type == commonMemType::hostOnly) {
		//allocate mem
		hostPtr = new unsigned char[size];
		//copy data
		for (size_t i = 0; i < noElements; ++i) {
			((T*)hostPtr)[i] = ((T*)other.hostPtr)[i];
		}
	}
	else if (type == commonMemType::deviceOnly) {
		//allocate mem
		cudaMalloc(&devicePtr, size);
		//copy data
		cudaMemcpy(devicePtr, other.devicePtr, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	}
	else {
		//host side

		//allocate mem
		hostPtr = new unsigned char[size];
		//copy data
		for (size_t i = 0; i < noElements; ++i) {
			((T*)hostPtr)[i] = ((T*)other.hostPtr)[i];
		}

		//device side

		//allocate mem
		cudaMalloc(&devicePtr, size);
		//copy data
		cudaMemcpy(devicePtr, other.devicePtr, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

	}
}


template <typename T>
commonMemory<T>::~commonMemory() {
	if (hostPtr != nullptr)delete[] hostPtr;
	if (devicePtr != nullptr)cudaFree(devicePtr);
}