#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rendering.cuh"

int Main() {

	std::cout << "hello world  1" << std::endl;
	return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	int x = 800, y = 600;
	window w1(hInstance, nCmdShow, L"hello world", x, y);

	camera c(vec3d(0,-1,0),x,y,vec3d(0,0,0),vec3d(1,0,0),vec3d(0,0,1));


	for (int i = 0; i < x * y; ++i) {
		w1.data[i * 3 + 2] = 255;
	}
	render(c,w1.data);
	w1.draw();
	
	
	system("pause");
	return 0;
}