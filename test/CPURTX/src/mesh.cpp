#include "mesh.h"


collTriangle::collTriangle(const triangle& t) {
	calc(t);
}


void collTriangle::calc(const triangle& t) {
	vec3d collNormal = vec3d::cross(*(t.pts[1]) - *(t.pts[0]), *(t.pts[2]) - *(t.pts[0]));
	collPlane.set(*(t.pts[0]), collNormal);
	for (int i = 0; i < 3; ++i) {
		sidePlanes[i].setPT(*t.pts[i]);
		sidePlanes[i].setDR(vec3d::cross(collNormal, *t.pts[(i + 1) % 3] - *t.pts[i]));
	}
}