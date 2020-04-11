#include <iostream>

#include "win32WindowingSystem.h"

int Main() {

	std::cout << "hello world  1" << std::endl;
	return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	//enableConsole();
	
	{
		window w1(hInstance, nCmdShow, L"hello world", 800, 600);
	}
	system("pause");
	return 0;
}