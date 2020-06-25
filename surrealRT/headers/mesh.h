#pragma once

#include "vec3.cuh"

struct mesh {
	vec3f pts[3];
};

struct meshConstrained {
	vec3f planeNormal;//can be vec3f
	vec3f coordCalcData;// 1/b.n, b.a, 1/a^2        b = p[2]-p[0]
	vec3f sn;//side normal , perpendicular to a
	vec3f a;// p[1]-p[0]
	
	//obsolete original method using side planes		
	//vec3f sidePlaneNormals[3];//can be vec3f
};