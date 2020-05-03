#pragma once
#include <Windows.h>
#include "linearMath.cuh"
#include "mesh.h"

typedef vec3f color;//x=r , y=g , z=b

struct colorBYTE {
	BYTE b, g, r;
};

struct intersectionParam {
	double lambda;
	vec3d pt;
};

struct fragmentProperties {
	short camX, camY;
	linearMathD::line ray;
	intersectionParam ip;
};

class chromaticShader {
public:
	__device__ chromaticShader() {}
	__device__ ~chromaticShader(){}
	__device__ virtual color shade(fragmentProperties& sd) { return color{ 0,0,0 }; }
};

class solidColor : public chromaticShader {
public:
	color c;
	__device__ solidColor() { c.x = 0; c.y = 0; c.z = 0; }
	__device__ solidColor(color C) { c = C; }
	__device__ ~solidColor() {}
	__device__ color shade(fragmentProperties& sd) { return c; }
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
	__device__ color shade(fragmentProperties& sd) {
		float m = sd.ray.getDr().mag();
		float ratio = sd.ray.getDr().z/m;
		ratio += 1;
		ratio /= 2;
		color rVal;
		rVal = up * ratio + down * (1 - ratio);
		ratio = sd.ray.getDr().x / m;
		ratio += 1;
		ratio /= 2;
		rVal += xp * ratio + xn * (1 - ratio);
		ratio = sd.ray.getDr().y / m;
		ratio += 1;
		ratio /= 2;
		rVal += yp * ratio + yn * (1 - ratio);
		return rVal;
	}
};

struct meshShaded {
	mesh M;
	chromaticShader* colShader;
};
