#include "parsing/classes/polynomialMesh.h"

#include "linearEqnSolver.h"
#include <iostream>

polynomialMesh::polynomialMesh(std::vector<vec3d>& vertices){
	unsigned long int size = vertices.size();
	xCoeff = size / 3;
	yCoeff = size / 3;
	zCoeff = size / 3;
	if (size % 3 > 0)xCoeff++;
	if (size % 3 > 1)yCoeff++;

	//size++;
	
	double *arr = new double[size * (size + 1)];// x = size+1 , y = size
	//for (unsigned long int i = 0; i < size - 1; ++i)arr[i] = 0;
	for (unsigned long int i = 0; i < size; ++i) {
		//arr[i * (size + 1) + size - 1] = 1;
		arr[i * (size + 1) + size] = -1;
	}
	//arr[size] = -10;

	for (unsigned long int i = 1; i < size; ++i) {

		double temp = 1;
		for (unsigned int j = 0; j < xCoeff; ++j) {
			temp *= vertices[i].x;
			arr[i * (size + 1) + j] = temp;
		}

		temp = 1;
		for (unsigned int j = 0; j < yCoeff; ++j) {
			temp *= vertices[i].y;
			arr[i * (size + 1) + j + xCoeff] = temp;
		}

		temp = 1;
		for (unsigned int j = 0; j < zCoeff; ++j) {
			temp *= vertices[i].z;
			arr[i * (size + 1) + j + xCoeff + yCoeff] = temp;
		}
	}

	LES::system sys(0, 0);
	sys.load(size, size, arr);
	sys.displayMatrix(std::cout);
	int notFound = sys.solve(true , 0);
	std::cout << "solved : "<<notFound<<"coefficients extra \n\n";
	sys.displayMatrix(std::cout);

	std::cout << "solution : \n\n";
	LES::system::solType type = sys.getSolution(coeff);
	for (unsigned long int i = 0; i < coeff.size(); ++i)std::cout << coeff[i] << " , ";
	std::cout << std::endl;
	if (type == LES::system::solType::inconsistant)std::cout << "inconsistant \n";
	if (type == LES::system::solType::uniqueSol)std::cout << "unique \n";
	if (type == LES::system::solType::infiniteSols)std::cout << "infinite \n";
	delete[] arr;
}


void polynomialMesh::displayEqn(std::ostream& f) {
	if (coeff.size() < xCoeff + yCoeff + zCoeff )return;
	f << std::fixed;
	for (unsigned int j = 0; j < xCoeff; ++j) {
		f << "(" <<coeff[j] << " * x^" << j+1 << ") + ";
	}
	for (unsigned int j = yCoeff; j < xCoeff+yCoeff; ++j) {
		f << "(" << coeff[j] << " * y^" << j+1-yCoeff << ") + ";
	}
	for (unsigned int j = xCoeff+yCoeff; j < xCoeff + yCoeff+zCoeff; ++j) {
		f << "(" << coeff[j] << " * z^" << j+1-(xCoeff + yCoeff) << ") + ";
	}
	//f << coeff[coeff.size() - 1]<<" = 0";
	f << "0 = 0";
	f << std::scientific;

}