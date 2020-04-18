#pragma once
#include "linearMath.h"

struct triangle {
	vec3d* pts[3];
};


//can be modified
class colltriangle {
public:

	linearMathD::plane collPlane;
	vec3d s[2];
};

