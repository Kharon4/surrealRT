#pragma once
#define math3D_DeclrationOnly 1
#include "chromaticShader.cuh"
#include "cudaRelated/commonMemory.cuh"

#include <string>

commonMemory<meshShaded> loadModel(std::string fileNameWithExtension, chromaticShader* shader);
