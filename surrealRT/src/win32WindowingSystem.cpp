#include "win32WindowingSystem.h"

#include <iostream>
#include <thread>
#include<chrono>
#include<map>

#include<gdiplus.h>
#include<gdiplusheaders.h>



std::map <HWND, window*> handleWindowMap;

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


class Initializer {
public:
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	Initializer() {
		Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
		for (int i = 0; i < 256; ++i) {
				input::isDown[i] = false;
				input::pressed[i] = false;
				input::released[i] = false;
				input::go[i] = false;
				input::end[i] = false;
		}
		input::update();
		input::update();
	}
	~Initializer() {
		Gdiplus::GdiplusShutdown(gdiplusToken);
	}
} globlalInitializere;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void createWindowInternal(HINSTANCE hInstance, int nCmdShow, window* val , HWND* out,bool *closed) {
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
	handleWindowMap.insert(std::pair<HWND, window*>(hwnd, val));

	MSG msg = { };

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	*closed = true;
}

window::window(HINSTANCE hInstance, int nCmdShow, LPCWSTR title, short X, short Y) {
	x = X;
	y = Y;
	Title = title;
	data = new BYTE[x * y * 3];

	dHBMP = nullptr;
	sHBMP = nullptr;

	dHBMP = &hbmp0;
	sHBMP = &hbmp1;
	std::thread t(createWindowInternal, hInstance, nCmdShow, this, &windowHandle,&closed);
	t.detach();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){

	switch (uMsg){
	case WM_DESTROY:
		handleWindowMap.erase(handleWindowMap.find(hwnd));
		PostQuitMessage(0);
		return 0;
	case WM_PAINT:
		handleWindowMap.find(hwnd)->second->draw();
		return 0;
	}

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

window::~window() {
	SendMessage(windowHandle, WM_DESTROY,0,0);
	delete[] data;
	DeleteObject(*dHBMP);
	DeleteObject(*sHBMP);
}


void window::draw() {
	HDC hdc;
	hdc = GetDC(windowHandle);//get DC for drawing

	//set drawing mode for fast BitBlt
	Gdiplus::Graphics graphics(hdc, windowHandle);
	graphics.SetCompositingMode(Gdiplus::CompositingMode::CompositingModeSourceCopy);
	graphics.SetCompositingQuality(Gdiplus::CompositingQuality::CompositingQualityHighSpeed);
	graphics.SetPixelOffsetMode(Gdiplus::PixelOffsetMode::PixelOffsetModeNone);
	graphics.SetSmoothingMode(Gdiplus::SmoothingMode::SmoothingModeNone);
	graphics.SetInterpolationMode(Gdiplus::InterpolationMode::InterpolationModeDefault);

	//get compatible DC for window 
	HDC hdcMem = CreateCompatibleDC(hdc);
	
	drawLock.lock();
	//converts the hBitmap into the correct form
	HBITMAP hbmOld = (HBITMAP)SelectObject(hdcMem, *sHBMP);//save old contents
	
	//store the data about the hBitmap into a structure
	BITMAP bm;
	GetObject(*sHBMP, sizeof(bm), &bm);

	//BitBlt
	BitBlt(hdc, 0, 0, bm.bmWidth, bm.bmHeight, hdcMem, 0, 0, SRCCOPY);

	//restore old content
	SelectObject(hdcMem, hbmOld);
	drawLock.unlock();

	//delete compatible DC
	DeleteDC(hdcMem);

	//draw completed relaease window DC
	ReleaseDC(windowHandle, hdc);
}

void window::update() {
	//bitmap with the data
	Gdiplus::Bitmap bmp(x, y, 4 * ((x * 24 + 31) / 32), PixelFormat24bppRGB, data);//
	
	//delete hBitmap
	DeleteObject(*dHBMP);//
	//get hBitmap from bitmap
	Gdiplus::Color backgroundCol(0, 0, 0);//
	bmp.GetHBITMAP(backgroundCol, dHBMP);//

	//invert d and s HBMP
	drawLock.lock();
	HBITMAP* tempPtr = sHBMP;
	sHBMP = dHBMP;
	dHBMP = tempPtr;
	drawLock.unlock();
	draw();
}

POINT window::GlobalToScreen(POINT global) { ScreenToClient(windowHandle, &global); return global; }
POINT window::ScreenToGlobal(POINT screen) { ClientToScreen(windowHandle, &screen); return screen; }

bool window::isWindowClosed() { return closed; }

unsigned long long input::millis(){
	uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	return ms;
}

unsigned long long input::micros() {
	uint64_t micro = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	return micro;
}

bool input::released[256];
bool input::isDown[256];
bool input::pressed[256];
bool input::go[256];
bool input::end[256];

unsigned long long input::time;

long int input::lockX, input::lockY;
long int input::mouseX, input::mouseY;
long int input::changeX, input::changeY;

bool input::lock = false;
double input::deltaTime = 0;

input::input() {
	for (int i = 0; i < 256; ++i) {
		isDown[i] = false;
		pressed[i] = false;
		released[i] = false;
		go[i] = false;
		end[i] = false;
	}
	time = millis();
}

void input::hideCursor() {
	(ShowCursor(FALSE));
}

void input::showCursor() {
	ShowCursor(TRUE);
}

void input::update() {
	//update keyboard stuff

	for (int i = 0; i < 256; ++i) {
		bool temp = GetAsyncKeyState(i);
		//go
		if (temp && isDown[i])go[i] = true;
		else go[i] = false;
		//end
		if (!temp && !isDown[i])end[i] = true;
		else end[i] = false;

		isDown[i] = temp;



		if (!pressed[i] && isDown[i] && !go[i]) {
			pressed[i] = true;
		}
		else {
			pressed[i] = false;
		}

		if (!released[i] && !isDown[i] && !end[i]) {
			released[i] = true;
		}
		else {
			released[i] = false;
		}

	}

	//update mouse stuff

	POINT p;
	GetCursorPos(&p);
	changeX = p.x - mouseX;
	changeY = p.y - mouseY;
	mouseX = p.x;
	mouseY = p.y;
	
	if (lock) {
		SetCursorPos(lockX, lockY);
		GetCursorPos(&p);
		mouseX = p.x;
		mouseY = p.y;
	}

	//get deltatime
	long long temp = millis();
	time = temp - time;
	deltaTime = time / 1000.0;
	time = temp;
}