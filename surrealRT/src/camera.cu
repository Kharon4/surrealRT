#include "camera.cuh"

screen::screen(short xRes, short yRes, vec3f ScCenter, vec3f hfRight, vec3f hfUp) {
	resX = xRes;
	resY = yRes;
	screenCenter = ScCenter;
	halfRight = hfRight;
	halfUp = hfUp;
}


camera::camera(vec3f Vertex ,short xRes, short yRes, vec3f ScCenter, vec3f hfRight, vec3f hfUp) {
	vertex = Vertex;
	sc = screen(xRes, yRes, ScCenter, hfRight, hfUp);
}