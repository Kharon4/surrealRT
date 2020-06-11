#pragma once
#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rendering.cuh"
#include "parsing/parsingAlgos/obj.h"
#include "parsing/classes/texture.h"

#define noTestCases 100

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	int x = 800, y = 800;
	window w1(hInstance, nCmdShow, L"surrealRT", x, y);
	

	//load the 2 textures initially 
	texture IMG[2] = {texture("res/kharon4.png", commonMemType::both),texture("res/kharon4_1.png", commonMemType::both) };


	for (int i = 0; i < noTestCases; ++i) {
		std::cout << "test " << i << " started ..................  ";
		texture tex = IMG[0];
		tex.copyToBuffer((colorBYTE*)w1.data, x, y);
		w1.update();
		
		texture tex1=IMG[1];
		tex1.copyToBuffer((colorBYTE*)w1.data, x, y);
		w1.update();
		//copy tests
		
		//replacement of default constructor
		texture T = tex1;
		
		//move constructor
		T = texture(900, 900);

		//copy constructor
		T = tex;
		T.copyToBuffer((colorBYTE*)w1.data, x, y);
		w1.update();



		std::cout << "successfully completed \n";

	}


	w1.update();
	system("pause");
	return 0;
}
