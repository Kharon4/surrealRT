#pragma once

#include "vec3.cuh"

struct mesh {
	vec3f pts[3];
};

struct meshConstrained {
	vec3f planeNormal;//can be vec3f
	vec3f sidePlaneNormals[3];//can be vec3f
};