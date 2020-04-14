#pragma once

#include "chromaticShader.cuh"

struct mesh {
	vec3d pts[3];
	chromaticShader* colShader;
};