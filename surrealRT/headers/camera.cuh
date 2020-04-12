#pragma once
#include "vec3.cuh"

class screen {
public:
	vec3d screenCenter, halfRight, halfUp;
	short resX, resY;

	screen(short xRes = 800, short yRes = 600, vec3d ScCenter = vec3d(0, 0, 0), vec3d hfRight = vec3d(1, 0, 0), vec3d hfUp = vec3d(0, 0, 1));
};

class camera {
public:
	screen sc;
	vec3d vertex;
	camera(vec3d Vertex, short xRes, short yRes, vec3d ScCenter, vec3d hfRight, vec3d hfUp);
};