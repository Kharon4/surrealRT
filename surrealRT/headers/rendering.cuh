#pragma once

#include "chromaticShader.cuh"
#include "camera.cuh"
#include "linearMath.cuh"
#include "rotation.cuh"
#include "cudaRelated/commonMemory.cuh"


void render(camera cam,BYTE *data);

class graphicalWorld {
private:
	commonMemory<meshShaded>* mesh;
	commonMemory<meshConstrained>* meshC;
public:
	graphicalWorld(commonMemory<meshShaded>* meshPtr) { 
		mesh = meshPtr; 
		meshC = new commonMemory<meshConstrained>(mesh->getNoElements(), commonMemType::deviceOnly); 
	}

	void render(camera cam, BYTE* data);

	~graphicalWorld() { delete meshC; }
};
