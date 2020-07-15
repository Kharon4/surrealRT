#pragma once
#define math3D_DeclrationOnly 1
#include "chromaticShader.cuh"
#include "cudaRelated/commonMemory.cuh"

#include <string>
#include <vector>

enum class loadAxisExchange : unsigned char
{
	xyz = 0, // inhouse scheme
	xzy = 1, // blender
	yxz = 2,
	yzx = 3,
	zxy = 4,
	zyx = 5
};

//returns commonMem allocated on bots host & device
commonMemory<meshShaded> loadModel(std::string fileNameWithExtension, chromaticShader* shader, loadAxisExchange vertexAxis = loadAxisExchange::xyz);

//host only function
//returns no meshShaded loaded , if -ve , function failed , the -ve val is baffMaxSize - Actaul size
long int loadModel(meshShaded* OUTBuff, unsigned int buffMaxSize, std::string fileNameWithExtension, chromaticShader* shader, loadAxisExchange vertexAxis = loadAxisExchange::xyz);

void loadBlankModel(meshShaded* OUTBuff, unsigned int buffMaxSize);//dissable shader

void loadModelVertices(std::vector<vec3d>& OUTdata, std::istream& f, loadAxisExchange vertexAxis = loadAxisExchange::xyz);