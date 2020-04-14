#include "rendering.cuh"
#include <iostream>

#define threadNo 1024
#define blockNo(Threads) Threads/threadNo


__global__
void initRays(short xRes , short yRes , vec3d vertex , vec3d topLeft , vec3d right , vec3d down , linearMathD::line * rays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= (xRes * yRes))return;
	
	short x, short y;
	x = tId % xRes;
	y = tId / yRes;

	rays[tId].setRaw_s(vertex, vec3d::subtract(vec3d::add(topLeft, vec3d::add(vec3d::multiply(right, (x + 0.5) / xRes), vec3d::multiply(down, (y + 0.5) / yRes))), vertex));
}

__global__
void getIntersections(linearMathD::line * rays , mesh ** intersections , size_t noRays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noRays)return;
	intersections[tId] = nullptr;
}

__global__
void shadeKernel(mesh** interactions, color* data, chromaticShader** defaultShader , size_t maxNo) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= maxNo)return;
	shaderData df;
	if (interactions[tId] == nullptr)data[tId] = (*defaultShader)->shade(df);
}

__global__
void createShader(chromaticShader ** ptr){
	color c;
	c.r = 0;
	c.g = 100;
	c.b = 200;
	*ptr = new solidColor(c);
	//ptr->c.r = 0;
	//ptr->c.g = 100;
	//ptr->c.b = 200;
}

__global__
void deleteShader(chromaticShader** ptr)
{
	delete (*ptr);
}


void render(camera cam,BYTE *data) {
	linearMathD::line * rays;
	cudaMalloc(&rays, sizeof(linearMathD::line) * cam.sc.resX * cam.sc.resY);
	//std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	initRays<<<threadNo , blockNo(cam.sc.resX*cam.sc.resY)>>>(cam.sc.resX, cam.sc.resY, cam.vertex, cam.sc.screenCenter - cam.sc.halfRight + cam.sc.halfUp, cam.sc.halfRight * 2, cam.sc.halfUp * -2, rays);
	//std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	mesh** intersections;
	cudaMalloc(&intersections, sizeof(mesh*) * cam.sc.resX * cam.sc.resY);
	//std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	getIntersections << <threadNo, blockNo(cam.sc.resX * cam.sc.resY) >> > (rays, intersections, cam.sc.resX * cam.sc.resY);
	//std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	chromaticShader** sc;
	cudaMalloc(&sc, sizeof(chromaticShader*));
	createShader<<<1,1>>>(sc);
	color* Data;
	cudaMalloc(&Data, sizeof(color) * cam.sc.resX * cam.sc.resY);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	shadeKernel << <threadNo, blockNo(cam.sc.resX * cam.sc.resY) >> > (intersections, Data, sc, cam.sc.resX * cam.sc.resY);

	cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	deleteShader<<<1,1>>>(sc);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(cudaMemcpy(data, Data, sizeof(color) * cam.sc.resX * cam.sc.resY, cudaMemcpyKind::cudaMemcpyDeviceToHost))<<std::endl;
	cudaDeviceSynchronize();
	cudaFree(Data);
	cudaFree(sc);
	cudaFree(intersections);
	cudaFree(rays);
}