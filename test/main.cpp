#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rotation.h"

bool updateCam(manipulation3dD::transform& t, manipulation3dD::transform& rOnly) {
	float rSpeed = -0.1;
	
	input::update();
	if (input::isDown['X'])return false;
	if (input::pressed['M']) {
		input::lock = !input::lock;
		input::lockX = input::mouseX;
		input::lockY = input::mouseY;
	}

	if (input::lock) {
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
	window w1(hInstance, nCmdShow, L"hello world", x, y);
	for (int i = 0; i < x * y; ++i) {
		w1.data[i * 3 + 2] = 255;
	}
	w1.draw();

	system("pause");
	
	
	
	/*
	camera c(vec3d(0, -1, 0), x, y, vec3d(0, 0, 0), vec3d(1, 0, 0), vec3d(0, 0, 1));
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
	while (updateCam(t,tDr)) {
		render(c, w1.data);
		w1.draw();
	}*/
	
	
	return 0;
}