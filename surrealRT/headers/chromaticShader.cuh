#pragma once
#include <Windows.h>

#include "vec3.cuh"

typedef vec3f color;//x=r , y=g , z=b

struct colorBYTE {
	BYTE b, g, r;
};

struct shaderMask {
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
	__device__ chromaticShader() {
		sm.camCoord = false;
		sm.surfaceNormal = false;
	}
	__device__ ~chromaticShader(){}
	__device__ virtual color shade(shaderData& sd) { return color{ 0,0,0 }; }
};

class solidColor : public chromaticShader {
public:
	color c;
	__device__ solidColor() { c.x = 0; c.y = 0; c.z = 0; }
	__device__ solidColor(color C) { c = C; }
	__device__ ~solidColor() {}
	__device__ color shade(shaderData& sd) { return c; }
};


class skybox :public chromaticShader {
public:
	color up;
	color down;
	color xp;
	color xn;
	color yp;
	color yn;
	__device__ skybox(color Up, color Down, color XP, color XN, color YP, color YN) { up = Up; down = Down; xp = XP; xn = XN; yp = YP; yn = YN; }
	__device__ skybox(){}
	__device__ color shade(shaderData& sd) {
		float m = sd.dr.mag();
		float ratio = sd.dr.z/m;
		ratio += 1;
		ratio /= 2;
		color rVal;
		rVal = up * ratio + down * (1 - ratio);
		ratio = sd.dr.x / m;
		ratio += 1;
		ratio /= 2;
		rVal += xp * ratio + xn * (1 - ratio);
		ratio = sd.dr.y / m;
		ratio += 1;
		ratio /= 2;
		rVal += yp * ratio + yn * (1 - ratio);
		return rVal;
	}
};
