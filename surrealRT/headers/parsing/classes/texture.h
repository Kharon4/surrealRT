#pragma once
#include "color.h"
#include "cudaRelated/commonMemory.cuh"
#include <string>


class texture {
private:
public:
	texture(unsigned short x , unsigned short y);
	texture(std::string fileName);

	
	~texture();
};