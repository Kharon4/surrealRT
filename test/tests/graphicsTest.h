#pragma once
#include <iostream>

#include "win32WindowingSystem.h"
#include "rendering.cuh"
#include "parsing/parsingAlgos/obj.h"
#include "parsing/classes/texture.h"
#include "transform.cuh"

bool updateCam(manipulation3d::transformf& t, manipulation3d::transformf& rOnly, bool & fpsDebug) {
	float rSpeed = -0.05;
	float mSpeed = 2;

	input::update();
	if (input::isDown['X'])return false;
	if (input::pressed['F'])fpsDebug = (!fpsDebug);
	if (input::pressed['M']) {
		input::lock = !input::lock;
		if (input::lock);
		else input::showCursor();
		input::lockX = input::mouseX;
		input::lockY = input::mouseY;
	}

	vec3f displacement(0, 0, 0);

	if (input::isDown[0xA0])mSpeed *= 10;

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
	int x = 512, y = 512;
	texture tex("res/kharon4_1.png", commonMemType::hostOnly);
	window w1(hInstance, nCmdShow, L"surrealRT", x, y);
	tex.copyToBuffer((colorBYTE*)w1.data, x, y);
	w1.update();
	camera c(vec3f(0, -1, 0), x, y, vec3f(0, 0, 0), vec3f(1, 0, 0), vec3f(0, 0, ((float)y) / x));
	manipulation3d::transformf t, tDr;
	t.CS.setOrigin(c.vertex);
	t.CS.setScale(vec3f(1, 1, 1));
	t.CS.setAngle(vec3f(math3D_pi / 2, 0, 0));
	tDr.CS.setScale(vec3f(1, 1, 1));
	tDr.CS.setAngle(vec3f(math3D_pi / 2, 0, 0));

	t.addVec(c.vertex, &c.vertex);
	t.addVec(c.sc.screenCenter, &c.sc.screenCenter);
	tDr.addVec(c.sc.halfRight, &c.sc.halfRight);
	tDr.addVec(c.sc.halfUp, &c.sc.halfUp);
	color testColor,lightCol;
	testColor = vec3f(100, 100, 100);
	lightCol = vec3f(1, 1, 2.5);
	//shadedSolidColCPU col(testColor, lightCol, vec3f(1, 2, -3));
	//randomTriangleShaderCPU col(vec3f(255,255,255), vec3f(7,17,18));
	texture tex2("res/triangleTexturing.png", commonMemType::both);
	textureShaderCPU col(tex2.getDevicePtr(), tex2.getWidth(), tex2.getHeight(), 0, 0, tex2.getWidth(), 0, 0, tex2.getHeight(),1);
	std::cout << "hello\n";
	commonMemory<meshShaded> temp(1500);
	loadBlankModel(temp.getHost(), temp.getNoElements());
	long int facesLoaded = loadModel(temp.getHost(),temp.getNoElements(),"res/monkey.obj", col.getGPUPtr(), loadAxisExchange::xzy);
	facesLoaded += loadModel(temp.getHost() + facesLoaded, temp.getNoElements() - facesLoaded, "res/cube.obj", col.getGPUPtr(),loadAxisExchange::xzy);
	//loaded
	std::cout << "no faces loaded = " << facesLoaded<<" / "<<temp.getNoElements() << std::endl;
	
	//graphicalWorld world(&temp);
	graphicalWorldADV world(&temp, x, y,3,3);

	input::asyncGetch();

	bool fpsOut = false;
	while (updateCam(t, tDr,fpsOut) && (!w1.isWindowClosed())) {
		unsigned long long start, renderTime, drawTime;
		start = input::micros();
		
		///multithreader
		world.render(c, w1.data, [&w1]() {w1.update(); });
		
		//single threaded
		//world.render(c, w1.data);
		renderTime = input::micros();
		
		//w1.update();
		
		drawTime = input::micros();
		if(fpsOut)
		std::cout << 1000000.0 / (drawTime - start) << "    Render time = " << renderTime - start <<"    draw time = " << drawTime - renderTime << std::endl;
	}


	return 0;
}