#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rendering.cuh"

bool updateCam(manipulation3dD::transform& t, manipulation3dD::transform& rOnly) {
	float rSpeed = -0.05;
	float mSpeed = 1;

	input::update();
	if (input::isDown['X'])return false;
	if (input::pressed['M']) {
		input::lock = !input::lock;
		if (input::lock);
		else input::showCursor();
		input::lockX = input::mouseX;
		input::lockY = input::mouseY;
	}

	vec3d displacement(0,0,0);

	if (input::isDown['W'])
		displacement += vec3d(1, 0, 0);
	if (input::isDown['S'])
		displacement -= vec3d(1, 0, 0);
	if (input::isDown['D'])
		displacement -= vec3d(0, 1, 0);
	if (input::isDown['A'])
		displacement += vec3d(0, 1, 0);
	if (input::isDown['E'])
		displacement += vec3d(0, 0, 1);
	if (input::isDown['Q'])
		displacement -= vec3d(0, 0, 1);

	displacement.normalize();

	t.CS.addRelativePos(displacement * mSpeed * input::deltaTime);


	if (input::lock) {
		input::hideCursor();
		t.CS.setAngle(t.CS.getAngle() + vec3d(rSpeed * input::changeX, rSpeed * input::changeY, 0)*input::deltaTime);
		rOnly.CS.setAngle(t.CS.getAngle() + vec3d(rSpeed * input::changeX, rSpeed * input::changeY, 0)*input::deltaTime);
	}
	t.update();
	rOnly.update();
	return true;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {

	enableConsole();
	int x = 800, y = 600;
	window w1(hInstance, nCmdShow, L"test window", x, y);
	for (int i = 0; i < x * y; ++i) {
		w1.data[i * 3 + 0] = 70;
		w1.data[i * 3 + 1] = 0;
		w1.data[i * 3 + 2] = 70;
	}
	w1.update();


	camera c(vec3d(0, -1, 0), x, y, vec3d(0, 0, 0), vec3d(1, 0, 0), vec3d(0, 0, ((float)y)/x));
	manipulation3dD::transform t , tDr;
	t.CS.setOrigin(c.vertex);
	t.CS.setScale(vec3d(1, 1, 1));
	t.CS.setAngle(vec3d(pi/2,0,0));
	tDr.CS.setScale(vec3d(1, 1, 1));
	tDr.CS.setAngle(vec3d(pi / 2, 0, 0));

	t.addVec(c.vertex, &c.vertex);
	t.addVec(c.sc.screenCenter, &c.sc.screenCenter);
	tDr.addVec(c.sc.halfRight, &c.sc.halfRight);
	tDr.addVec(c.sc.halfUp, &c.sc.halfUp);
	commonMemory<meshShaded> temp(2);
	temp.getHost()[0].M.pts[0] = vec3d(-1, 0, -1);
	temp.getHost()[0].M.pts[1] = vec3d(1, 0, -1);
	temp.getHost()[0].M.pts[2] = vec3d(1, 2, -1);
	temp.getHost()[1].M.pts[0] = vec3d(-1, 0, -1);
	temp.getHost()[1].M.pts[1] = vec3d(1, 2, -1);
	temp.getHost()[1].M.pts[2] = vec3d(-1, 2, -1);
	color testColor;
	testColor.z = 255;
	solidColCPU col(testColor);
	temp.getHost()[0].colShader = col.getGPUPtr();
	temp.getHost()[1].colShader = col.getGPUPtr();
	
	graphicalWorld world(&temp);
	
	while (updateCam(t,tDr)&& (!w1.isWindowClosed())) {
		world.render(c, w1.data);
		w1.update();
	}
	
	
	return 0;
}