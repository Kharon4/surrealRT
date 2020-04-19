#pragma once
#include "linearMath.h"

struct triangle {
	vec3d* pts[3];
	vec3f reflectivity;
	vec3f transminivity;
	vec3f diffuseRefelctivity;
	float refractiveIndex;
};


//can be modified
class collTriangle {
public:
	collTriangle() {}
	collTriangle(const triangle & t);
	void calc(const triangle& t);
	linearMathD::plane collPlane;
	linearMathD::plane sidePlanes[3];
};

