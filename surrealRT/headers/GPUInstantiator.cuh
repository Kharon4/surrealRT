#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GlobalGPUDefsConstants.cuh"

#ifdef __NVCC__
template <typename parentClass, typename childClass, typename... args>
__global__
void GPUCreateInstance(parentClass** mem, args... constructorArgs) {
	*mem = new childClass(constructorArgs...);
}


template <typename parentClass>
__global__
void GPUDeleteInstance(parentClass** mem) {
	delete(*mem);
}


template <typename parentClass, typename childClass, typename... args>
class CPUInstanceController {
private:
	parentClass* ptrToCClass = nullptr;
public:
	CPUInstanceController(args... constructorArgs) {
		parentClass** ptrToPtrToCClass = nullptr;
		cudaMalloc(&ptrToPtrToCClass, sizeof(parentClass*));
		cudaDebugDeviceSync();
		GPUCreateInstance<parentClass, childClass, args...> << <1, 1 >> > (ptrToPtrToCClass, constructorArgs...);
		cudaDebugDeviceSync();
		cudaMemcpy(&ptrToCClass, ptrToPtrToCClass, sizeof(parentClass*), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaDebugDeviceSync();
		cudaFree(ptrToPtrToCClass);
	}

	parentClass* getGPUPtr() { return ptrToCClass; }

	~CPUInstanceController() {
		parentClass** ptrToPtrToCClass = nullptr;
		cudaMalloc(&ptrToPtrToCClass, sizeof(parentClass*));
		cudaDebugDeviceSync();
		cudaMemcpy(ptrToPtrToCClass,&ptrToCClass, sizeof(parentClass*), cudaMemcpyKind::cudaMemcpyHostToDevice);
		cudaDebugDeviceSync();
		GPUDeleteInstance<parentClass> << <1, 1 >> > (ptrToPtrToCClass);
		cudaDebugDeviceSync();
		cudaFree(ptrToPtrToCClass);
	}
};

#else

template <typename parentClass, typename childClass, typename... args>
class CPUInstanceController {
private:
	parentClass** ptrToPtrToCClass = nullptr;
	parentClass* ptrToCClass = nullptr;
public:
	CPUInstanceController(args... constructorArgs);
	parentClass* getGPUPtr();
	~CPUInstanceController();
};

#endif // __NVCC__

