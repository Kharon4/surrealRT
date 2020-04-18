#include "rendering.cuh"
#include <iostream>

#define threadNo 1024
#define blockNo(Threads) Threads/threadNo

struct intersectionParam{
	short camX, camY;
	double lambda;
	vec3d pt;
	mesh* mesh;
	meshConstrained* meshConstrained;
};


__global__
void initRays(short xRes , short yRes , vec3d vertex , vec3d topLeft , vec3d right , vec3d down , linearMathD::line * rays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= (xRes * yRes))return;
	
	short x, short y;
	x = tId % xRes;
	y = tId / yRes;

	rays[tId].setRaw_s(vertex, vec3d::subtract(vec3d::add(topLeft, vec3d::add(vec3d::multiply(right, (x + 0.5) / xRes), vec3d::multiply(down, (y + 0.5) / yRes))), vertex));
}

__device__ __host__ void calculateMeshConstraints(mesh* Mesh , meshConstrained *meshC){
	vec3d plNormal = vec3d::cross(Mesh->pts[1] - Mesh->pts[0], Mesh->pts[2] - Mesh->pts[0]);
	meshC->planeNormal = plNormal;
	meshC->sidePlaneNormals[0] = vec3d::cross(plNormal, Mesh->pts[1] - Mesh->pts[0]);
	meshC->sidePlaneNormals[1] = vec3d::cross(plNormal, Mesh->pts[2] - Mesh->pts[1]);
	meshC->sidePlaneNormals[2] = vec3d::cross(plNormal, Mesh->pts[0] - Mesh->pts[2]);
}

__global__
void initMesh(mesh* Mesh, meshConstrained* meshC, size_t noOfThreads) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noOfThreads)return;
	calculateMeshConstraints(Mesh + tId, meshC + tId);
}

__global__
void getIntersections(linearMathD::line * rays , mesh ** intersections , size_t noRays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noRays)return;
	intersections[tId] = nullptr;
}

__global__
void shadeKernel(mesh** interactions,linearMathD::line* rays, color* data, chromaticShader** defaultShader , size_t maxNo) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= maxNo)return;
	shaderData df;
	df.dr = rays[tId].getDr();
	if (interactions[tId] == nullptr)data[tId] = (*defaultShader)->shade(df);
}

__global__
void getByteColor(color* data, colorBYTE* dataByte, float min, float delta, size_t noThreads) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noThreads)return;
	color rval = data[tId];
	rval -= vec3f(min, min, min);
	rval *= 256 / delta;
	if (rval.x > 255)dataByte[tId].r = 255;
	else if (rval.x < 0)dataByte[tId].r = 0;
	else dataByte[tId].r = (unsigned char)rval.x;
	if (rval.y > 255)dataByte[tId].g = 255;
	else if (rval.y < 0)dataByte[tId].g = 0;
	else dataByte[tId].g = (unsigned char)rval.y;
	if (rval.z > 255)dataByte[tId].b = 255;
	else if (rval.z < 0)dataByte[tId].b = 0;
	else dataByte[tId].b = (unsigned char)rval.z;
}

__global__
void createShader(chromaticShader ** ptr){
	color c,down,red;
	c.x = -255;
	c.y = 100;
	c.z = 200;
	down.x = -255;
	down.y = 0;
	down.z = 0;
	red.x = 700;
	*ptr = new skybox(c,down,red,down,down,down);
}

__global__
void deleteShader(chromaticShader** ptr)
{
	delete (*ptr);
}

void displayCudaError() {
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
}

void render(camera cam,BYTE *data) {
	linearMathD::line * rays;
	cudaMalloc(&rays, sizeof(linearMathD::line) * cam.sc.resX * cam.sc.resY);
	initRays<<<threadNo , blockNo(cam.sc.resX*cam.sc.resY)>>>(cam.sc.resX, cam.sc.resY, cam.vertex, cam.sc.screenCenter - cam.sc.halfRight + cam.sc.halfUp, cam.sc.halfRight * 2, cam.sc.halfUp * -2, rays);
	mesh** intersections;
	cudaMalloc(&intersections, sizeof(mesh*) * cam.sc.resX * cam.sc.resY);
	getIntersections << <threadNo, blockNo(cam.sc.resX * cam.sc.resY) >> > (rays, intersections, cam.sc.resX * cam.sc.resY);
	chromaticShader** sc;
	cudaMalloc(&sc, sizeof(chromaticShader*));
	createShader<<<1,1>>>(sc);
	color* Data;
	cudaMalloc(&Data, sizeof(color) * cam.sc.resX * cam.sc.resY);
	shadeKernel << <threadNo, blockNo(cam.sc.resX * cam.sc.resY) >> > (intersections,rays, Data, sc, cam.sc.resX * cam.sc.resY);
	colorBYTE *DataByte;
	cudaMalloc(&DataByte, sizeof(colorBYTE) * cam.sc.resX * cam.sc.resY);
	getByteColor << <threadNo, blockNo(cam.sc.resX * cam.sc.resY) >> > (Data, DataByte, 0, 256, cam.sc.resX * cam.sc.resY);
	cudaMemcpy(data, DataByte, sizeof(colorBYTE) * cam.sc.resX * cam.sc.resY, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(DataByte);
	cudaFree(Data);
	deleteShader << <1, 1 >> > (sc);
	cudaFree(sc);
	cudaFree(intersections);
	cudaFree(rays);
}