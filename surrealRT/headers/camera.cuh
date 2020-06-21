#pragma once

#include "vec3.cuh"

class screen {
public:
	vec3f screenCenter, halfRight, halfUp;
	short resX, resY;

	screen(short xRes = 800, short yRes = 600, vec3f ScCenter = vec3f(0, 0, 0), vec3f hfRight = vec3f(1, 0, 0), vec3f hfUp = vec3f(0, 0, 1));
};

class camera {
public:
	screen sc;
	vec3f vertex;
	camera(vec3f Vertex, short xRes, short yRes, vec3f ScCenter, vec3f hfRight, vec3f hfUp);
};