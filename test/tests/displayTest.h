#pragma once
#include <iostream>

#define math3D_DeclrationOnly 1

#include "win32WindowingSystem.h"
#include "rendering.cuh"
#include "parsing/parsingAlgos/obj.h"
#include "parsing/classes/texture.h"

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	int x = 1081 , y = 877;
	window w1(hInstance, nCmdShow, L"surrealRT", x, y);
	for (int i = 0; i < y; ++i) {
		for (int j = 0; j < x; ++j) {
			w1.data[4 * (i * x + j) + 1] = 25*(j % 10);
			//w1.data[3 * (i * x + j) + 1] = 50*(i % 4);
			//w1.data[3 * (i * x + j) + 2] = 25*(i % 8);
		}
	}
	
	w1.update();
	system("pause");
	return 0;
}