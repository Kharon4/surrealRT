#pragma once
#include <Windows.h>
#include "linearMath.cuh"
#include "mesh.h"
#include "GPUInstantiator.cuh"
#include "color.h"


struct intersectionParam {
	double lambda;
	vec3d pt;
	meshConstrained* MC;
	mesh* M;
};

struct fragmentProperties {
	short camX, camY;
	linearMathD::line *ray;
	intersectionParam ip;
};

class chromaticShader {
public:
	__device__ chromaticShader() {}
	__device__ ~chromaticShader(){}
	__device__ virtual color shade(fragmentProperties& sd) { return color{ 0,0,0 }; }
};



//more types

class solidColor : public chromaticShader {
public:
	color c;
	__device__ solidColor() { c.x = 0; c.y = 0; c.z = 0; }
	__device__ solidColor(color C) { c = C; }
	__device__ ~solidColor() {}
	__device__ color shade(fragmentProperties& sd) { return c; }
};

typedef CPUInstanceController<chromaticShader, solidColor, color> solidColCPU;

#ifdef __NVCC__
int fSolicCol() {
	color c;
	solidColCPU shader(c);
	shader.getGPUPtr();
	return c.x;
}
#endif // __NVCC__


class shadedSolidColor : public chromaticShader {
public:
	color c;
	color light;
	vec3d dir;


	__device__ shadedSolidColor(color C, color Light, vec3d DIR) { c = C; dir = vec3d::normalizeRaw_s(DIR); light = Light; }
	__device__ ~shadedSolidColor(){}

	__device__ color shade(fragmentProperties& sd) {
		vec3f rVal = (light * vec3d::dot(sd.ip.MC->planeNormal, dir));
		rVal.x *= c.x;
		rVal.y *= c.y;
		rVal.z *= c.z;
		return (rVal + c);
	}
};

typedef CPUInstanceController<chromaticShader, shadedSolidColor, color, color, vec3d> shadedSolidColCPU;

#ifdef __NVCC__
int fShadedSolicCol() {
	color c;
	shadedSolidColCPU shader(c,c,vec3d(0,0,0));
	shader.getGPUPtr();
	return c.x;
}
#endif // __NVCC__


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
		float m = sd.ray->getDr().mag();
		float ratio = sd.ray->getDr().z/m;
		ratio += 1;
		ratio /= 2;
		color rVal;
		rVal = up * ratio + down * (1 - ratio);
		ratio = sd.ray->getDr().x / m;
		ratio += 1;
		ratio /= 2;
		rVal += xp * ratio + xn * (1 - ratio);
		ratio = sd.ray->getDr().y / m;
		ratio += 1;
		ratio /= 2;
		rVal += yp * ratio + yn * (1 - ratio);
		return rVal;
	}
};

typedef CPUInstanceController<chromaticShader, skybox, color, color, color, color, color, color> skyboxCPU;
#ifdef __NVCC__
int fSkybox() {
	color c;
	skyboxCPU shader(c, c, c, c, c, c);
	shader.getGPUPtr();
	return c.x;
}
#endif // __NVCC__


struct meshShaded {
	mesh M;
	chromaticShader* colShader = nullptr;
};
