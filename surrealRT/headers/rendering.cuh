#pragma once

#include "chromaticShader.cuh"
#include "camera.cuh"
#include "linearMath.cuh"
#include "rotation.cuh"
#include "cudaRelated/commonMemory.cuh"

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

	void renderPartial(camera cam);
	void copyData(camera cam,BYTE* data);
	~graphicalWorld() { delete meshC; }
};
