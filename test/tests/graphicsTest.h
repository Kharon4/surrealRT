#pragma once
#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rendering.cuh"
#include "parsing/parsingAlgos/obj.h"
#include "parsing/classes/texture.h"


bool updateCam(manipulation3dF::transform& t, manipulation3dF::transform& rOnly) {
	float rSpeed = -0.05;
	float mSpeed = 2;

	input::update();
	if (input::isDown['X'])return false;
	if (input::pressed['M']) {
		input::lock = !input::lock;
		if (input::lock);
		else input::showCursor();
		input::lockX = input::mouseX;
		input::lockY = input::mouseY;
	}

	vec3f displacement(0, 0, 0);

	if (input::isDown['W'])
		displacement += vec3f(1, 0, 0);
	if (input::isDown['S'])
		displacement -= vec3f(1, 0, 0);
	if (input::isDown['D'])
		displacement -= vec3f(0, 1, 0);
	if (input::isDown['A'])
		displacement += vec3f(0, 1, 0);
	if (input::isDown['E'])
		displacement += vec3f(0, 0, 1);
	if (input::isDown['Q'])
		displacement -= vec3f(0, 0, 1);

	displacement.normalize();

	t.CS.addRelativePos(displacement * mSpeed * input::deltaTime);


	if (input::lock) {
		input::hideCursor();
		t.CS.setAngle(t.CS.getAngle() + vec3f(rSpeed * input::changeX, rSpeed * input::changeY, 0) * input::deltaTime);
		rOnly.CS.setAngle(t.CS.getAngle() + vec3f(rSpeed * input::changeX, rSpeed * input::changeY, 0) * input::deltaTime);
	}
	t.update();
	rOnly.update();
	return true;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	int x = 1080, y = 720;
	texture tex("res/kharon4.png", commonMemType::both);
	window w1(hInstance, nCmdShow, L"surrealRT", x, y);
	tex.copyToBuffer((colorBYTE*)w1.data, x, y);
	w1.update();
	input::asyncGetch();
	camera c(vec3f(0, -1, 0), x, y, vec3f(0, 0, 0), vec3f(1, 0, 0), vec3f(0, 0, ((float)y) / x));
	manipulation3dF::transform t, tDr;
	t.CS.setOrigin(c.vertex);
	t.CS.setScale(vec3f(1, 1, 1));
	t.CS.setAngle(vec3f(pi / 2, 0, 0));
	tDr.CS.setScale(vec3f(1, 1, 1));
	tDr.CS.setAngle(vec3f(pi / 2, 0, 0));

	t.addVec(c.vertex, &c.vertex);
	t.addVec(c.sc.screenCenter, &c.sc.screenCenter);
	tDr.addVec(c.sc.halfRight, &c.sc.halfRight);
	tDr.addVec(c.sc.halfUp, &c.sc.halfUp);
	color testColor;
	testColor = vec3f(100, 100, 100);
	shadedSolidColCPU col(testColor, testColor / 500, vec3f(1, 2, 3));
	std::cout << "hello\n";
	std::cout << "size of loadAxisExchange = " << sizeof(loadAxisExchange) << std::endl;
	commonMemory<meshShaded> temp = loadModel("res/icoSphere.obj", col.getGPUPtr(),loadAxisExchange::xzy);
	//loaded
	std::cout << "no faces loaded = " << temp.getNoElements() << std::endl;
	graphicalWorld world(&temp);
	

	while (updateCam(t, tDr) && (!w1.isWindowClosed())) {
		unsigned long long start, uTime;
		start = input::micros();
		world.render(c, w1.data, [&w1]() {w1.update(); });
		uTime = input::micros();
		std::cout << 1000000.0 / (uTime - start) << std::endl;
	}


	return 0;
}