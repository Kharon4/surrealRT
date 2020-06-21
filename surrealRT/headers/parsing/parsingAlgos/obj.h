#pragma once
#define math3D_DeclrationOnly 1
#include "chromaticShader.cuh"
#include "cudaRelated/commonMemory.cuh"

#include <string>

enum class loadAxisExchange : unsigned char
{
	xyz = 0, // inhouse scheme
	xzy = 1, // blender
	yxz = 2,
	yzx = 3,
	zxy = 4,
	zyx = 5
};

commonMemory<meshShaded> loadModel(std::string fileNameWithExtension, chromaticShader* shader, loadAxisExchange vertexAxis = loadAxisExchange::xyz);
