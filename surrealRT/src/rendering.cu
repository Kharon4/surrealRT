#include "rendering.cuh"

#include <thread>

#ifdef __GPUDEBUG
#include <iostream>
#endif

#define threadNo 1024
#define blockNo(Threads) ((Threads/threadNo) + 1)

__global__
void initRays(short xRes , short yRes , vec3f vertex , vec3f topLeft , vec3f right , vec3f down , linearMath::linef * rays) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= (xRes * yRes))return;
	
	short x, short y;
	x = tId % xRes;
	y = tId / xRes;

	rays[tId].setRaw_s(vertex, vec3f::subtract(vec3f::add(topLeft, vec3f::add(vec3f::multiply(right, (x + 0.5) / xRes), vec3f::multiply(down, (y + 0.5) / yRes))), vertex));
}

__device__ __host__ void calculateMeshConstraints(mesh* Mesh , meshConstrained *meshC){
	vec3f plNormal = vec3f::cross(Mesh->pts[1] - Mesh->pts[0], Mesh->pts[2] - Mesh->pts[0]);
	meshC->planeNormal = vec3f::normalizeRaw_s(plNormal);
	meshC->sidePlaneNormals[0] = vec3f::normalizeRaw_s(vec3f::cross(plNormal, Mesh->pts[1] - Mesh->pts[0]));
	meshC->sidePlaneNormals[1] = vec3f::normalizeRaw_s(vec3f::cross(plNormal, Mesh->pts[2] - Mesh->pts[1]));
	meshC->sidePlaneNormals[2] = vec3f::normalizeRaw_s(vec3f::cross(plNormal, Mesh->pts[0] - Mesh->pts[2]));
}

__global__
void initMesh(meshShaded* Mesh, meshConstrained* meshC, size_t noOfThreads) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noOfThreads)return;
	calculateMeshConstraints(&((Mesh + tId)->M), meshC + tId);
}

__device__ __host__
void getClosestIntersection(meshShaded * Mesh ,meshConstrained* meshC, linearMath::linef ray, size_t noTrs,  meshShaded * &OUTmesh, meshConstrained * &OUTMeshC , float& OUTlambda, vec3f& OUTpt) {
	OUTmesh = nullptr;
	OUTMeshC = nullptr;
	OUTlambda = -1;
	double tempDist;
	for (size_t i = 0; i < noTrs; ++i) {
		
		switch (Mesh[i].colShader->meshVProp)
		{
		case (meshVisibilityProperties::inActive):
			tempDist = -1;
			break;
		case (meshVisibilityProperties::frontActive):
			{	
				float dotCalculated = vec3f::dot(ray.getDr(), meshC[i].planeNormal);
				if (dotCalculated > 0) {
					tempDist = vec3f::dot(Mesh[i].M.pts[0] - ray.getPt(), meshC[i].planeNormal) / dotCalculated;
				}
				else tempDist = -1;
			}
			break;

		case (meshVisibilityProperties::backActive):
			{
				float dotCalculated = vec3f::dot(ray.getDr(), meshC[i].planeNormal);
				if (dotCalculated < 0) {
					tempDist = vec3f::dot(Mesh[i].M.pts[0] - ray.getPt(), meshC[i].planeNormal) / dotCalculated;
				}
				else tempDist = -1;
			}
				break;
		case (meshVisibilityProperties::frontBackActive):
			{
				float dotCalculated = vec3f::dot(ray.getDr(), meshC[i].planeNormal);
				if (dotCalculated != 0) {
					tempDist = vec3f::dot(Mesh[i].M.pts[0] - ray.getPt(), meshC[i].planeNormal) / dotCalculated;
				}
				else tempDist = -1;
			}
			break;

		}


		//check for visibility
		if (tempDist > 0 && (tempDist < OUTlambda || OUTlambda < 0)) {
			//check for inside
			vec3f pt = linearMath::getPt(ray, tempDist);
			if (vec3f::dot(pt - Mesh[i].M.pts[0], meshC[i].sidePlaneNormals[0]) < 0)continue;
			if (vec3f::dot(pt - Mesh[i].M.pts[1], meshC[i].sidePlaneNormals[1]) < 0)continue;
			if (vec3f::dot(pt - Mesh[i].M.pts[2], meshC[i].sidePlaneNormals[2]) < 0)continue;
			//inside
			OUTlambda = tempDist;

			OUTpt = pt;
			OUTmesh = Mesh + i;
			OUTMeshC = meshC + i;
		}
	}
}

__global__
void getIntersections(linearMath::linef* rays, size_t noRays, meshShaded* trs, meshConstrained* collTrs, size_t noTrs, color* displayData, chromaticShader* defaultShader) {
	size_t tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId >= noRays)return;

	fragmentProperties fp;
	fp.ray = rays + tId;
	meshShaded* outM;
	getClosestIntersection(trs, collTrs, *fp.ray, noTrs, outM, fp.ip.MC, fp.ip.lambda, fp.ip.pt);

	//shade
	if (outM == nullptr) {
		fp.ip.M = nullptr;
		displayData[tId] = (defaultShader)->shade(fp);
	}
	else {
		fp.ip.M = &(outM->M);
		displayData[tId] = outM->colShader->shade(fp);
	}
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

void displayCudaError(size_t id = 0) {
#ifdef __GPUDEBUG
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	std::cout << "#" << id <<"  "<< cudaGetErrorName(err)<< std::endl;
	if (err != cudaError::cudaSuccess) {
		int x;
		std::cin >> x;
	}
#else

#endif
}

void generateGPUDisplatData(colorBYTE** data , camera cam) {
	displayCudaError(11);
	cudaMalloc(data, sizeof(colorBYTE) * cam.sc.resX * cam.sc.resY);
	displayCudaError(10);
}

void renderIntermediate(camera cam,colorBYTE* DataByte, meshShaded* meshS, meshConstrained* meshC, size_t noTrs) {
	displayCudaError(9);
	linearMath::linef* rays;
	displayCudaError(8);
	cudaMalloc(&rays, sizeof(linearMath::linef) * cam.sc.resX * cam.sc.resY);
	displayCudaError(7);
	initRays << <blockNo(cam.sc.resX * cam.sc.resY), threadNo >> > (cam.sc.resX, cam.sc.resY, cam.vertex, cam.sc.screenCenter - cam.sc.halfRight + cam.sc.halfUp, cam.sc.halfRight * 2, cam.sc.halfUp * -2, rays);
	displayCudaError(1);
	skyboxCPU defaultShader(color(0, 0, 128), color(-200, -200, -200), color(150, 0, 0), color(0, 0, 64), color(0, 0, 64), color(0, 0, 64));
	//solidColCPU defaultShader(color(0, 0, 0));
	displayCudaError(2);
	color* Data;
	cudaMalloc(&Data, sizeof(color) * cam.sc.resX * cam.sc.resY);
	displayCudaError(3);
	getIntersections << <blockNo(cam.sc.resX * cam.sc.resY), threadNo >> > (rays, cam.sc.resX * cam.sc.resY, meshS, meshC, noTrs, Data, defaultShader.getGPUPtr());
	displayCudaError(4);
	getByteColor << <blockNo(cam.sc.resX * cam.sc.resY), threadNo >> > (Data, DataByte, 0, 256, cam.sc.resX * cam.sc.resY);
	displayCudaError(5);
	cudaFree(Data);
	cudaFree(rays);
	displayCudaError(6);
}

void cpyData(colorBYTE* data , BYTE * displayData, camera cam) {
	cudaDeviceSynchronize();
	displayCudaError(12);
	cudaMemcpy(displayData, data, sizeof(colorBYTE) * cam.sc.resX * cam.sc.resY, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	displayCudaError(13);
	cudaFree(data);
	displayCudaError(14);
}

void Render(camera cam,BYTE *data, meshShaded * meshS , meshConstrained * meshC , size_t noTrs) {
	colorBYTE* displayData;
	generateGPUDisplatData(&displayData, cam);
	renderIntermediate(cam, displayData, meshS, meshC, noTrs);
	cpyData(displayData, data, cam);
}


void graphicalWorld::render(camera cam, BYTE* data) {


	bool updated=false;
	meshShaded* devPtr = meshS->getDevice(&updated);
	if(updated){
		initMesh<<<blockNo(meshS->getNoElements()),threadNo>>>(devPtr, meshC->getDevice(), meshS->getNoElements());
	}
	Render(cam, data, meshS->getDevice(), meshC->getDevice(), meshS->getNoElements());

}

void graphicalWorld::render(camera cam, BYTE* data, std::function<void()> drawCall) {
	std::thread draw(drawCall);
	renderPartial(cam);
	draw.join();
	copyData(cam, data);
}

void graphicalWorld::renderPartial(camera cam) {
	
	bool updated = false;
	meshShaded* devPtr = meshS->getDevice(&updated);
	if (updated) {
		initMesh << <blockNo(meshS->getNoElements()), threadNo >> > (devPtr, meshC->getDevice(), meshS->getNoElements());
	}
	if (tempData != nullptr) {
		cudaDeviceSynchronize();
		cudaFree(tempData);
		tempData = nullptr;
		displayCudaError(16);
	}
	generateGPUDisplatData(&tempData, cam);
	renderIntermediate(cam, tempData, meshS->getDevice(), meshC->getDevice(), meshS->getNoElements());
	//Render(cam, data, meshS->getDevice(), meshC->getDevice(), meshS->getNoElements());
}

void graphicalWorld::copyData(camera cam, BYTE* data) {
	if (tempData != nullptr) {
		cpyData(tempData, data, cam);
		tempData = nullptr;
	}
}