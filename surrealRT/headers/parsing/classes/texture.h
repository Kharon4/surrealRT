#pragma once
#include "color.h"
#include "cudaRelated/commonMemory.cuh"
#include <string>


#pragma comment (lib,"Gdiplus.lib")

class texture {
private:
	unsigned short x, y;

	commonMemory<colorBYTE>* Data = nullptr;
public:
	texture(unsigned short X , unsigned short Y, commonMemType type = commonMemType::both);
	texture(std::string fileName, commonMemType type = commonMemType::both);

	unsigned short getWidth();
	unsigned short getHeight();
	
	colorBYTE* getDevicePtr();
	colorBYTE* getHostPtr();

	void copyToBuffer(colorBYTE* data);//host only

	~texture();
};