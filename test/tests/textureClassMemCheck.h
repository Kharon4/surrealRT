#pragma once
#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rendering.cuh"
#include "parsing/parsingAlgos/obj.h"
#include "parsing/classes/texture.h"

#define noTestCases 1024

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	int x = 800, y = 800;
	window w1(hInstance, nCmdShow, L"surrealRT", x, y);
	
	for (int i = 0; i < noTestCases; ++i) {
		std::cout << "test " << i << " started ..................  ";
		texture tex("res/kharon4.png", commonMemType::both);
		tex.copyToBuffer((colorBYTE*)w1.data, x, y);
		w1.update();
		
		texture tex1("res/kharon4_1.png", commonMemType::both);
		tex1.copyToBuffer((colorBYTE*)w1.data, x, y);
		w1.update();
		std::cout << "successfully completed \n";
	}


	w1.update();
	system("pause");
	return 0;
}
