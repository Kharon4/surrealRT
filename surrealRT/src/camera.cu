#include "camera.cuh"

screen::screen(short xRes, short yRes, vec3d ScCenter, vec3d hfRight, vec3d hfUp) {
	resX = xRes;
	resY = yRes;
	screenCenter = ScCenter;
	halfRight = hfRight;
	halfUp = hfUp;
}


camera::camera(vec3d Vertex ,short xRes, short yRes, vec3d ScCenter, vec3d hfRight, vec3d hfUp) {
	vertex = Vertex;
	sc = screen(xRes, yRes, ScCenter, hfRight, hfUp);
}