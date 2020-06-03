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


	//host only
	void copyToBuffer(colorBYTE* data);
	void copyToBuffer(colorBYTE* data, unsigned short width, unsigned short height);//scale

	//host only crop
	void copyToBufferCrop(color* data, unsigned short xL, unsigned short yL, unsigned short xM, unsigned short yM);


	~texture();
};