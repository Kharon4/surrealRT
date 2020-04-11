#include <iostream>

#include "win32WindowingSystem.h"

int Main() {

	std::cout << "hello world  1" << std::endl;
	return 0;
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
	return 0;
}