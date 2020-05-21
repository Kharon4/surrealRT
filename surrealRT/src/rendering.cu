#include "rendering.cuh"
#include <iostream>

#define threadNo 32
#define blockNo(Threads) ((Threads/threadNo) + 1)


#define _enableDebug 0

__global__
void initRays(short xRes , short yRes , vec3d vertex , vec3d topLeft , vec3d right , vec3d down , linearMathD::line * rays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= (xRes * yRes))return;
	
	short x, short y;
	x = tId % xRes;
	y = tId / xRes;

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
void initMesh(meshShaded* Mesh, meshConstrained* meshC, size_t noOfThreads) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noOfThreads)return;
	calculateMeshConstraints(&((Mesh + tId)->M), meshC + tId);
}

__device__ __host__
void getClosestIntersection(meshShaded * Mesh ,meshConstrained* meshC, linearMathD::line ray, size_t noTrs,  meshShaded * &OUTmesh, meshConstrained * &OUTMeshC , double& OUTlambda, vec3d& OUTpt) {
	OUTmesh = nullptr;
	OUTMeshC = nullptr;
	OUTlambda = -1;
	double tempDist;
	for (size_t i = 0; i < noTrs; ++i) {
		linearMathD::intersectionLambdaRaw_s(ray, linearMathD::plane(Mesh[i].M.pts[0], meshC[i].planeNormal), tempDist);
		//check for visibility
		if (tempDist > 0 && (tempDist < OUTlambda || OUTlambda < 0)) {
			//check for inside
			vec3d pt = linearMathD::getPt(ray, tempDist);
			if (vec3d::dot(pt - Mesh[i].M.pts[0], meshC[i].sidePlaneNormals[0]) < 0)continue;
			if (vec3d::dot(pt - Mesh[i].M.pts[1], meshC[i].sidePlaneNormals[1]) < 0)continue;
			if (vec3d::dot(pt - Mesh[i].M.pts[2], meshC[i].sidePlaneNormals[2]) < 0)continue;
			//inside
			OUTlambda = tempDist;

			OUTpt = pt;
			OUTmesh = Mesh + i;
			OUTMeshC = meshC + i;
		}
	}
}

__global__
void getIntersections(linearMathD::line * rays, size_t noRays,meshShaded*trs , meshConstrained*collTrs, size_t noTrs,color* displayData, chromaticShader** defaultShader) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noRays)return;

	fragmentProperties fp;
	meshShaded *outM;
	meshConstrained* outCM;
	getClosestIntersection(trs, collTrs, rays[tId], noTrs,outM,outCM,fp.ip.lambda,fp.ip.pt);
	fp.ray = rays[tId];
	//shade
	if (outM == nullptr)displayData[tId] = (*defaultShader)->shade(fp);
	else displayData[tId] = outM->colShader->shade(fp);
}

/*
__device__
void shade(color* data, chromaticShader** defaultShader) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= maxNo)return;
	shaderData df;
	df.dr = rays[tId].getDr();
	if (interactions[tId] == nullptr)data[tId] = (*defaultShader)->shade(df);
}
*/

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

namespace dontAccess {
	__global__
	void createDefaultShader(chromaticShader** ptr) {
		color c;
		c.x = 100;
		c.y = 0;
		c.z = 100;
		*ptr = new solidColor(c);
	}
}


void displayCudaError() {
#if _enableDebug
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;
#else

#endif
}

void Render(camera cam,BYTE *data, meshShaded * meshS , meshConstrained * meshC , size_t noTrs) {
	linearMathD::line * rays;
	cudaMalloc(&rays, sizeof(linearMathD::line) * cam.sc.resX * cam.sc.resY);
	initRays<<<blockNo(cam.sc.resX*cam.sc.resY), threadNo >>>(cam.sc.resX, cam.sc.resY, cam.vertex, cam.sc.screenCenter - cam.sc.halfRight + cam.sc.halfUp, cam.sc.halfRight * 2, cam.sc.halfUp * -2, rays);
	displayCudaError();
	chromaticShader** sc;
	cudaMalloc(&sc, sizeof(chromaticShader*));
	createShader << <1, 1 >> > (sc);
	displayCudaError();
	color* Data;
	cudaMalloc(&Data, sizeof(color) * cam.sc.resX * cam.sc.resY);
	displayCudaError();
	getIntersections << <blockNo(cam.sc.resX * cam.sc.resY), threadNo >> > (rays, cam.sc.resX * cam.sc.resY,meshS,meshC,noTrs,Data,sc);
	displayCudaError();
	colorBYTE *DataByte;
	cudaMalloc(&DataByte, sizeof(colorBYTE) * cam.sc.resX * cam.sc.resY);
	getByteColor << <blockNo(cam.sc.resX * cam.sc.resY), threadNo >> > (Data, DataByte, 0, 256, cam.sc.resX * cam.sc.resY);
	displayCudaError();
	cudaMemcpy(data, DataByte, sizeof(colorBYTE) * cam.sc.resX * cam.sc.resY, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(DataByte);
	cudaFree(Data);
	deleteShader << <1, 1 >> > (sc);
	displayCudaError();
	cudaFree(sc);
	cudaFree(rays);
	displayCudaError();
}


void graphicalWorld::render(camera cam, BYTE* data) {
	bool updated=false;
	meshShaded* devPtr = meshS->getDevice(&updated);
	if(updated){
		initMesh<<<blockNo(meshS->getNoElements()),threadNo>>>(devPtr, meshC->getDevice(), meshS->getNoElements());
	}
	Render(cam, data, meshS->getDevice(), meshC->getDevice(), meshS->getNoElements());
}