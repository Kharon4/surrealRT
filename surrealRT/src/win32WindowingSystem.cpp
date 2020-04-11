#include "win32WindowingSystem.h"

#include <iostream>
#include <thread>

#include<gdiplus.h>
#include<gdiplusheaders.h>


//console functions
bool enableConsole() {
	if (!AllocConsole())return true;//error
	FILE* fDummy;
	freopen_s(&fDummy, "CONIN$", "r", stdin);
	freopen_s(&fDummy, "CONOUT$", "w", stderr);
	freopen_s(&fDummy, "CONOUT$", "w", stdout);
	std::cout.clear();
	std::clog.clear();
	std::cerr.clear();
	std::cin.clear();
	return false;
}

void disableConsole() {
	FreeConsole();
}

bool setConsoleState(bool enabled) {
	if (enabled)return enableConsole();
	disableConsole();
	return false;
}


class gdiInitializer {
public:
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	gdiInitializer() {
		Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	}
	~gdiInitializer() {
		Gdiplus::GdiplusShutdown(gdiplusToken);
	}
} globlaGDI;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void createWindowInternal(HINSTANCE hInstance, int nCmdShow, window* val , HWND* out) {
	// Register the window class.
	const wchar_t CLASS_NAME[] = L"Window Class";
	WNDCLASS wc = { };
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;

	RegisterClass(&wc);

	// Create the window.
	HWND hwnd = CreateWindowEx(
		0,                              // Optional window styles.
		CLASS_NAME,                     // Window class
		val->Title,    // Window text
		WS_OVERLAPPED|WS_SYSMENU,            // Window style
		// Size and position
		CW_USEDEFAULT, CW_USEDEFAULT, val->x, val->y,
		NULL,       // Parent window    
		NULL,       // Menu
		hInstance,  // Instance handle
		NULL        // Additional application data
	);

	if (hwnd == NULL)
	{
		return;
	}
	*out = hwnd;
	ShowWindow(hwnd, nCmdShow);
	MSG msg = { };

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	
}

window::window(HINSTANCE hInstance, int nCmdShow, LPCWSTR title, short X, short Y) {
	x = X;
	y = Y;
	Title = title;
	std::thread t(createWindowInternal, hInstance, nCmdShow, this, &windowHandle);
	t.detach();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){

	switch (uMsg){
	case WM_DESTROY:
		PostQuitMessage(0);
	}

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

window::~window() {
	SendMessage(windowHandle, WM_DESTROY,0,0);
}