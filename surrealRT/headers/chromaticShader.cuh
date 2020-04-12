#pragma once
#include <Windows.h>
#include "vec3.cuh"

struct color {
	BYTE b, g, r;
};

struct shaderMask {
	bool dr;//dr of ray
	bool pt;//pt of contact
	bool camCoord;//camera coordinates
	bool surfaceNormal;//normal of contact surface
};

struct shaderData {
	vec3d dr;
	vec3d pt;
	vec3s camCoord;
	vec3d surfaceNormal;
};

class chromaticShader {
public:
	shaderMask sm;
	__device__ chromaticShader();
	__device__ ~chromaticShader(){}
	__device__ virtual color shade(shaderData& sd) {}
};

class solidColor : public chromaticShader {
public:
	color c;
	__device__ solidColor(color C) { c = C; }
	__device__ ~solidColor();
	__device__ color shade(shaderData& sd) { return c; }
};

