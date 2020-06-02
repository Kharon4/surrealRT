#include "parsing/classes/texture.h"

#include <Windows.h>
#include <gdiplus.h>
#include <gdiplusheaders.h>

texture::texture(unsigned short X, unsigned short Y, commonMemType type) {
	x = X;
	y = Y;

	Data = new commonMemory<colorBYTE>(((size_t)x)*y, type);
}

texture::texture(std::string fileName, commonMemType type = commonMemType::both) {
	 
}


unsigned short texture::getWidth() { return x; }
unsigned short texture::getHeight() { return y; }

colorBYTE* texture::getDevicePtr() {
	return (Data->getDevice());
}

colorBYTE* texture::getHostPtr() {
	return (Data->getHost());
}


void texture::copyToBuffer(colorBYTE* data) {
	size_t size = x * y;
	colorBYTE* hPtr = getHostPtr();

	for (size_t i = 0; i < x * y; ++i)data[i] = hPtr[i];
}


texture::~texture() {
	delete Data;
}
