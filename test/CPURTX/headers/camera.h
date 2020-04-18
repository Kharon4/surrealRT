#pragma once

#include"linearMath.h"

struct camera {
	short xRes, yRes;
	vec3d vertex;
	vec3d topLeftCorner;
	vec3d right;
	vec3d down;
};

linearMathD::line getRay(camera c, short x, short y);