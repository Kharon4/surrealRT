#include <iostream>
#include "win32WindowingSystem.h"
#include "cudaRelated/commonMemory.cuh"

#define noElements 1024 * 128
#define noGos 2024

//128 MBS

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	enableConsole();
	
	for (int i = 0; i < noGos; ++i) {

		std::cout << "running test : " << i;

		//allocate common Mem
		commonMemory<int> cMem(noElements, commonMemType::both);

		//get host and device ptrs

		int * hPtr = cMem.getHost();
		
		//put content in host ptr
		for (size_t j = 0; j < cMem.getNoElements(); j++) {
			hPtr[j] = j % 3;
		}
		
		//transfer to GPU
		int* dPtr = cMem.getDevice();

		//reset content of hPtr
		for (size_t j = 0; j < cMem.getNoElements(); j++) {
			hPtr[j] = 3;
		}

		//revert data
		hPtr = cMem.getHost();

		//check data
		for (size_t j = 0; j < cMem.getNoElements(); j++) {
			if (hPtr[j] != (j % 3)) {
				std::cout << "inconsistant result 0 \n";
				system("pause");
			}
		}


		//test switching

		//both to host
		cMem.changeMemType(commonMemType::hostOnly);

		if (cMem.getMemType() != commonMemType::hostOnly) {
			std::cout << "inconsistant result 1 \n";
			system("pause");
		}

		if (cMem.getDevice() != nullptr) {
			std::cout << "inconsistant result 2 \n";
			system("pause");
		}

		hPtr = cMem.getHost();

		if(hPtr == nullptr) {
			std::cout << "inconsistant result 3 \n";
			system("pause");
		}

		//reset content of hPtr
		for (size_t j = 0; j < cMem.getNoElements(); j++) {
			hPtr[j] = 4;
		}



		//host to both
		cMem.changeMemType(commonMemType::both);

		if (cMem.getMemType() != commonMemType::both) {
			std::cout << "inconsistant result 4 \n";
			system("pause");
		}

		if (cMem.getDevice() == nullptr) {
			std::cout << "inconsistant result 5 \n";
			system("pause");
		}

		hPtr = cMem.getHost();

		if (hPtr == nullptr) {
			std::cout << "inconsistant result 6 \n";
			system("pause");
		}

		//reset content of hPtr
		for (size_t j = 0; j < cMem.getNoElements(); j++) {
			hPtr[j] = 5;
		}

		//both to device

		cMem.changeMemType(commonMemType::deviceOnly);

		if (cMem.getMemType() != commonMemType::deviceOnly) {
				std::cout << "inconsistant result 7 \n";
				system("pause");
		}

		if (cMem.getDevice() == nullptr) {
			std::cout << "inconsistant result 8 \n";
			system("pause");
		}

		if (cMem.getHost() != nullptr) {
			std::cout << "inconsistant result 9 \n";
			system("pause");
		}

		//device to host
		cMem.changeMemType(commonMemType::hostOnly);

		if (cMem.getMemType() != commonMemType::hostOnly) {
			std::cout << "inconsistant result 10 \n";
			system("pause");
		}

		if (cMem.getDevice() != nullptr) {
			std::cout << "inconsistant result 11 \n";
			system("pause");
		}

		hPtr = cMem.getHost();

		if (hPtr == nullptr) {
			std::cout << "inconsistant result 12 \n";
			system("pause");
		}

		//check data
		for (size_t j = 0; j < cMem.getNoElements(); j++) {
			if (hPtr[j] != 5) {
				std::cout << "inconsistant result 13 \n";
				system("pause");
			}
		}

		//host to device
		cMem.changeMemType(commonMemType::deviceOnly);

		if (cMem.getMemType() != commonMemType::deviceOnly) {
			std::cout << "inconsistant result 14 \n";
			system("pause");
		}

		if (cMem.getHost() != nullptr) {
			std::cout << "inconsistant result 15 \n";
			system("pause");
		}

		if (cMem.getDevice() == nullptr) {
			std::cout << "inconsistant result 16 \n";
			system("pause");
		}




		std::cout << "    ... Test completed successfully\n";
	}


	std::cout << "testing finished\n";
	system("pause");
	

	return 0;
}