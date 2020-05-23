#pragma once


#define __GPUDEBUG 1 //comment to go int release mode

//define debugging code
#ifdef __NVCC__

#ifdef __GPUDEBUG

#define cudaDebugDeviceSync() cudaDeviceSynchronize()

#else

#define cudaDebugDeviceSync() ;

#endif // GPUDEBUG

#endif // __NVCC__
