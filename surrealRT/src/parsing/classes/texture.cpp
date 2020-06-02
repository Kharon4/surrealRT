#include "parsing/classes/texture.h"

#include <Windows.h>
#include <gdiplus.h>
#include <gdiplusheaders.h>

//string conversion
#include <codecvt>
#include <locale>


//testing
#include <iostream>



using namespace Gdiplus;


texture::texture(unsigned short X, unsigned short Y, commonMemType type) {
	x = X;
	y = Y;

	Data = new commonMemory<colorBYTE>(((size_t)x)*y, type);
}

using convert_t = std::codecvt_utf8<wchar_t>;
std::wstring_convert<convert_t, wchar_t> strconverter;

std::wstring to_wstring(std::string str)
{
	return strconverter.from_bytes(str);
}


texture::texture(std::string fileName, commonMemType type) {
	Gdiplus::Bitmap bmp(to_wstring(fileName).c_str());
	x = bmp.GetWidth();
	y = bmp.GetHeight();

	//create common Mem
	Data = new commonMemory<colorBYTE>(((size_t)x) * y, type);

	//save Data

	size_t size = (size_t)(x)*y;
	colorBYTE* hPtr = Data->getHost();
	Gdiplus::Rect rect;
	rect.X = 0;
	rect.Y = 0;
	rect.Width = x;
	rect.Height = y;

	BitmapData data;
	bmp.LockBits(&rect, ImageLockMode::ImageLockModeRead,PixelFormat32bppRGB,&data);

	//collect data
	for (size_t i = 0; i < size; ++i) {
		hPtr[i] = ((colorBYTE*)(data.Scan0))[i];
	}

	bmp.UnlockBits(&data);
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
	size_t size = (size_t)(x) * y;
	colorBYTE* hPtr = getHostPtr();

	for (size_t i = 0; i < size; ++i)data[i] = hPtr[i];
}

void texture::copyToBuffer(colorBYTE* data, unsigned short width, unsigned short height) {
	colorBYTE* hPtr = Data->getHost();
	for (unsigned int i = 0; i < height; ++i) {
		for (unsigned int j = 0; j < width; ++j) {
			data[i * width + j] = hPtr[((j * x) / width) + ((i * y) / height) * x];
		}
	}
}



texture::~texture() {
	delete Data;
}
