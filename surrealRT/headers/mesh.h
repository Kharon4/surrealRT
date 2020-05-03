#pragma once

#include "vec3.cuh"

struct mesh {
	vec3d pts[3];
};

struct meshConstrained {
	vec3d planeNormal;//can be vec3f
	vec3d sidePlaneNormals[3];//can be vec3f
};