#include "rendering.cuh"
#include "linearMath.cuh"

__global__
void initRays(short xRes , short yRes , vec3d vertex , vec3d topLeft , vec3d right , vec3d down , linearMathD::line * rays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= (xRes * yRes))return;
	
	short x, short y;
	x = tId % xRes;
	y = tId / yRes;

	rays[tId].setPT(vertex);
	rays[tId].setDRRaw_s(vec3d::subtract(vec3d::add(topLeft, vec3d::add(vec3d::multiply(right,(x+0.5)/xRes), vec3d::multiply(down,(y+0.5)/yRes))),vertex));
}

__global__
void getIntersections(linearMathD::line * rays , mesh ** intersections , size_t noRays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noRays)return;
	intersections[tId] = nullptr;
}

__global__
void shadeKernel(mesh** interactions, color* data, chromaticShader* defaultShader , size_t maxNo) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= maxNo)return;
	shaderData df;
	if (interactions[tId] == nullptr)data[tId] = defaultShader->shade(df);
}


void render(camera cam, BYTE* data) {

}