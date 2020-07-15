#pragma once
#include "parsing/classes/polynomialMesh.h"
#include "parsing/parsingAlgos/obj.h"
#include "win32WindowingSystem.h"
#include <iostream>
#include <fstream>

using namespace std;

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow){
	enableConsole();
	
	std::vector<vec3d> rVal;
	std::ifstream file("res/cube.obj");
	
	loadModelVertices(rVal, file);
	cout << rVal.size() << endl;
	polynomialMesh mesh(rVal);
	cout <<"eqn :"<< endl;
	mesh.displayEqn();
	cout << endl << endl;
	system("pause");
	return 0;
}