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

	//constructors
	texture(unsigned short X , unsigned short Y, commonMemType type = commonMemType::both);
	texture(std::string fileName, commonMemType type = commonMemType::both);
	
	void operator= (texture&& other);//shallow copy / move constructor
	void operator= (const texture& other);//deep copy / copy constructor
	
	texture(texture&& other);//shallow copy / move constructor
	texture(const texture& other);//deep copy / copy constructor

	unsigned short getWidth();
	unsigned short getHeight();
	
	colorBYTE* getDevicePtr();
	colorBYTE* getHostPtr();

	void changeTextureResidance(commonMemType type);

	//host only
	void copyToBuffer(colorBYTE* data);
	void copyToBuffer(colorBYTE* data, unsigned short width, unsigned short height);//scale

	//host only crop
	//xl,yl inclusive , xm,ym non-inclusive
	void copyToBufferCrop(colorBYTE* data, unsigned short xL, unsigned short yL, unsigned short xM, unsigned short yM);
	void copyToBufferCrop(colorBYTE* data, unsigned short xL, unsigned short yL, unsigned short xM, unsigned short yM,unsigned short width , unsigned short height);



	//destrucutor
	~texture();
};