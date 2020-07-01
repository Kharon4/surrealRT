#pragma once
#include <Windows.h>
#include "linearMath.cuh"
#include "mesh.h"
#include "GPUInstantiator.cuh"
#include "color.h"


struct intersectionParam {
	float lambda;
	vec3f pt;
	meshConstrained* MC;
	mesh* M;
};

struct fragmentProperties {
	short camX, camY;
	linearMath::line<float> *ray;
	intersectionParam ip;
};

enum class meshVisibilityProperties : signed char {
	inActive = 0,
	frontActive = -1,//dont make +ve
	frontBackActive = 2,
	backActive = 1//dont make -ve
};

#define noneColor color(0,0,0)
#define errorColor color(255,0,255)

class chromaticShader {
public:
	meshVisibilityProperties meshVProp;
	__device__ chromaticShader() { meshVProp = meshVisibilityProperties::frontActive; }
	__device__ ~chromaticShader(){}
	__device__ virtual color shade(fragmentProperties& sd) { return noneColor; }
};

//#define __NVCC__ //only for debugging

#ifdef __NVCC__
//unique name for function + initialization of shader object
#define DefineForCompilation(name, ...)\
void* _TEMP_FUNC_DEFINATION_FOR_COMPILATION##name(){\
__VA_ARGS__ \
return shader.getGPUPtr();\
}
#else
#define DefineForCompilation(name, ...) ;
#endif

//more types

class disableShader : public chromaticShader {
public:
	__device__ disableShader() { meshVProp = meshVisibilityProperties::inActive; }
	__device__ ~disableShader() {}
	__device__ virtual color shade(fragmentProperties& sd) { return errorColor; }
};

typedef CPUInstanceController<chromaticShader, disableShader> disableShaderCPU;

DefineForCompilation(disableShaderCPU, disableShaderCPU shader;)


class solidColor : public chromaticShader {
public:
	color c;
	__device__ solidColor() { c.x = 0; c.y = 0; c.z = 0; }
	__device__ solidColor(color C) { c = C; }
	__device__ ~solidColor() {}
	__device__ color shade(fragmentProperties& sd) { return c; }
};

typedef CPUInstanceController<chromaticShader, solidColor, color> solidColCPU;

DefineForCompilation(solidColCPU, color c; solidColCPU shader(c);)


class shadedSolidColor : public chromaticShader {
public:
	color c;
	color light;
	vec3f dir;


	__device__ shadedSolidColor(color C, color Light, vec3f DIR) { c = C; dir = vec3f::normalizeRaw_s(DIR); light = Light; }
	__device__ ~shadedSolidColor(){}

	__device__ color shade(fragmentProperties& sd) {
		vec3f rVal = (light * vec3f::dot(sd.ip.MC->planeNormal, dir));
		rVal.x *= c.x;
		rVal.y *= c.y;
		rVal.z *= c.z;
		return (rVal + c);
	}
};

typedef CPUInstanceController<chromaticShader, shadedSolidColor, color, color, vec3f> shadedSolidColCPU;

DefineForCompilation(shadedSolidColCPU, color c; shadedSolidColCPU shader(c, c, vec3f(0,0,0));)


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

DefineForCompilation(skyboxCPU, color c; skyboxCPU shader(c, c, c, c, c, c);)



struct meshShaded {
	mesh M;
	chromaticShader* colShader = nullptr;
};
