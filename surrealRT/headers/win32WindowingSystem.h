#pragma once
#include <Windows.h>

bool enableConsole();
void disableConsole();
bool setConsoleState(bool enabled);


class window {
private:
	HWND windowHandle = NULL;
public:
	window(HINSTANCE hInstance, int nCmdShow, LPCWSTR title, short X, short Y);
	~window();
	short x, y;
	LPCWSTR Title;
};
