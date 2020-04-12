#include "chromaticShader.cuh"

__device__ chromaticShader::chromaticShader() {
	sm.camCoord = false;
	sm.dr = false;
	sm.pt = false;
	sm.surfaceNormal = false;
}