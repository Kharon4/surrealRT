#pragma once
#include <iostream>
#include <vector>
#include "vec3.cuh"

class polynomialMesh {
private:
	std::vector<double> coeff;
	unsigned int xCoeff = 0;
	unsigned int yCoeff = 0;
	unsigned int zCoeff = 0;
public:
	polynomialMesh(std::vector<vec3d>& vertices);
	void displayEqn(std::ostream& f = std::cout);
};