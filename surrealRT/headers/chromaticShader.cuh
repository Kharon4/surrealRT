#pragma once
#include <Windows.h>
#include "linearMath.cuh"
#include "mesh.h"
#include "GPUInstantiator.cuh"
#include "color.h"


struct intersectionParam {
	float lambda;
	float cx, cy;//local coordinates
	vec3f pt;
	meshConstrained* MC;
	mesh* M;
	unsigned int trId;
};

struct fragmentProperties {
	short camX, camY;
	linearMath::line<float> *ray;
	intersectionParam ip;
};

/*
enum class meshVisibilityProperties : signed char {
	inActive = 0,
	frontActive = -1,//dont make +ve
	frontBackActive = 2,
	backActive = 1//dont make -ve
};
*/

class meshVisibilityProperties {
private:
	unsigned char data = 0;
public:
	enum class bitNames :unsigned char {
		frontVisible = 0,
		backVisible = 1,
		triangle = 2 //if not considered parallelogram 
	};
	__device__ __host__ meshVisibilityProperties() { data = 0; }
	__device__ __host__ inline bool getBit(bitNames bit) { return (bool)(data & (1 << (unsigned char)bit)); }
	__device__ __host__ inline void setBit(bitNames bit) { data |= (1 << (unsigned char)bit); }
	__device__ __host__ inline void resetBit(bitNames bit) { data &= (~(1 << (unsigned char)bit)); }
	__device__ __host__ inline void toggleBit(bitNames bit){ data ^= (1 << (unsigned char)bit);}
};

#define noneColor color(0,0,0)
#define errorColor color(255,0,255)

class chromaticShader {
public:
	struct fragmentPropertiesMask {
		bool globalCoord = false;
		bool localCoord = false;
	} shaderMask;
	meshVisibilityProperties meshVProp;
	__device__ chromaticShader() {	meshVProp.setBit(meshVisibilityProperties::bitNames::frontVisible);
									meshVProp.setBit(meshVisibilityProperties::bitNames::triangle);}
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
	__device__ disableShader() { meshVProp.resetBit(meshVisibilityProperties::bitNames::frontVisible); }
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


class textureShader :public chromaticShader {
public:
	colorBYTE* data;
	short ox, oy, xx, xy, yx, yy;
	unsigned short textureWidth, textureHeight;
	float max;
	__device__ textureShader(colorBYTE* DATA,unsigned short Width,unsigned short Height, short Ox, short Oy, short Xx, short Xy, short Yx, short Yy,float Max) {
		shaderMask.localCoord = true;
		data = DATA;
		ox = Ox; oy = Oy; xx = Xx - Ox; xy = Xy - Oy; yx = Yx - Ox; yy = Yy - Oy;
		textureWidth = Width; textureHeight = Height; max = Max;
	}

	__device__ ~textureShader() {}

	__device__ color shade(fragmentProperties& sd) {
		int x = ox + xx * sd.ip.cx + yx * sd.ip.cy;
		int y = oy + yy * sd.ip.cy + xy * sd.ip.cx;

		if (x < 0)x = 0;
		else if(x >= textureWidth) x = textureWidth - 1;

		if (y < 0)y = 0;
		else if (y >= textureHeight)y = textureHeight - 1;
		colorBYTE cb = data[(y * textureWidth) + x];
		color c;
		c.x = cb.r * max;
		c.y = cb.g * max;
		c.z = cb.b * max;

		return c;
	}

};

typedef CPUInstanceController<chromaticShader, textureShader, colorBYTE*, unsigned short,unsigned  short, short, short, short, short, short, short,float> textureShaderCPU;

DefineForCompilation(textureShaderCPU, unsigned short ui = 15; short i = 0; float f = 1.1; colorBYTE data; textureShaderCPU shader(&data, ui, ui, i, i, i, i, i, i, f);)


class randomTriangleShader : public chromaticShader {
private:
	vec3f factors;
	vec3f range;
public:
#define randomLargeNo 172.871233
	__device__ randomTriangleShader(vec3f max, vec3f seed) { range = max; factors = seed; }
	__device__ ~randomTriangleShader() {}

#ifdef __NVCC__
	__device__ color shade(fragmentProperties& sd) {
		color c(sin((sd.ip.trId + factors.x)*randomLargeNo)
			,	sin((sd.ip.trId + factors.y)*randomLargeNo)
			,	sin((sd.ip.trId + factors.z)*randomLargeNo));

		c.x *= range.x;
		c.y *= range.y;
		c.z *= range.z;
		return c;
	}
#endif
};

typedef CPUInstanceController<chromaticShader, randomTriangleShader,vec3f,vec3f> randomTriangleShaderCPU;

DefineForCompilation(randomTriangleShaderCPU, vec3f tempf(0, 1, 2); randomTriangleShaderCPU shader(tempf, vec3f(1, 6, 4));)

struct meshShaded {
	mesh M;
	chromaticShader* colShader = nullptr;
};
