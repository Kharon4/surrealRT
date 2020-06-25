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

	short xBatch, yBatch;
	short xDoublingNo, yDoublingNo;

public:

	graphicalWorldADV(commonMemory<meshShaded>* meshPtr , short xBat , short yBat , short xI=1 , short yI=1) {
		meshS = meshPtr;
		meshC = new commonMemory<meshConstrained>(meshS->getNoElements(), commonMemType::deviceOnly);
		xBatch = xBat;
		yBatch = yBat;
		xDoublingNo = xI;
		yDoublingNo = yI;
	}

	void render(camera cam, BYTE* data);
	void render(camera cam, BYTE* data, std::function<void()> drawCall);

	~graphicalWorldADV() { delete meshC; }
};