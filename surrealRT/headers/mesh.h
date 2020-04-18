#pragma once

#include "chromaticShader.cuh"

struct mesh {
	vec3d pts[3];
	chromaticShader* colShader;
};

struct meshConstrained {
	vec3d planeNormal;//can be vec3f
	vec3d sidePlaneNormals[3];//can be vec3f
};