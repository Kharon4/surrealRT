#pragma once

#include "chromaticShader.cuh"
#include "camera.cuh"
#include "linearMath.cuh"
#include "rotation.cuh"
#include "cudaRelated/commonMemory.cuh"

#include <functional>

class graphicalWorld {
private:
	colorBYTE* tempData = nullptr;
	commonMemory<meshShaded>* meshS;
	commonMemory<meshConstrained>* meshC;
public:
	graphicalWorld(commonMemory<meshShaded>* meshPtr) { 
		meshS = meshPtr; 
		meshC = new commonMemory<meshConstrained>(meshS->getNoElements(), commonMemType::deviceOnly); 
	}



	void render(camera cam, BYTE* data);
	void render(camera cam, BYTE* data, std::function<void()> drawCall);


	void renderPartial(camera cam);
	void copyData(camera cam,BYTE* data);
	~graphicalWorld() { delete meshC; }
};


class graphicalWorldADV {// with assisted RTX
public:
	struct rayMeshData {
		meshShaded* M;
		meshConstrained Mc;
	};
private:

	colorBYTE* tempData = nullptr;
	commonMemory<meshShaded>* meshS;
	commonMemory<meshConstrained>* meshC;

	rayMeshData *redundancyData;

	unsigned short xResReq, yResReq;//res required for doubling to work
	unsigned short xDoublingIterations, yDoublingIterations;//doubling interations
	unsigned char xRes, yRes;//actual res
	unsigned short mulFacX, mulFacY;//2^xDI , 2^yDI

public:

	graphicalWorldADV(commonMemory<meshShaded>* meshPtr, unsigned short xResolution, unsigned short yRessolution, unsigned char xIters = 0, unsigned char yIters = 0);

	void render(camera cam, BYTE* data);
	void render(camera cam, BYTE* data, std::function<void()> drawCall);

	~graphicalWorldADV();
};