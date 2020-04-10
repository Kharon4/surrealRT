#include <iostream>

#include "win32WindowingSystem.h"

int Main() {

	std::cout << "hello world  1" << std::endl;
	return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	
	system("pause");
	return 0;
}