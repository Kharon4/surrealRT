#pragma once
#include <Windows.h>

bool enableConsole();
void disableConsole();
bool setConsoleState(bool enabled);

#pragma comment (lib,"Gdiplus.lib")

class window {
private:
	HWND windowHandle = NULL;
public:
	window(HINSTANCE hInstance, int nCmdShow, LPCWSTR title, short X, short Y);
	~window();
	short x, y;
	LPCWSTR Title;
};
