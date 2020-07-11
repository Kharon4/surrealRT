#pragma once
#include <Windows.h>
#include <mutex>


bool enableConsole();
void disableConsole();
bool setConsoleState(bool enabled);

#pragma comment (lib,"Gdiplus.lib")

class window {
private:
	HWND windowHandle = NULL;
	bool closed = false;

	std::mutex drawLock;

	HBITMAP hbmp0,hbmp1;

	HBITMAP* dHBMP;
	HBITMAP* sHBMP;

public:
	short x, y;
	LPCWSTR Title;
	BYTE* data;//BGR
	
	window(HINSTANCE hInstance, int nCmdShow, LPCWSTR title, short X, short Y);
	~window();

	void draw();//non time consuming

	void update(bool DRAW = false);//time consuming


	bool  isWindowClosed();

	POINT GlobalToScreen(POINT global);
	POINT ScreenToGlobal(POINT screen);
};

class input {
private:
	input();
	static unsigned long long time;
public:

	input(const input&) = delete;
	input& operator=(const input&) = delete;

	static bool isDown[256];
	static bool pressed[256];
	static bool released[256];
	static bool go[256];//start a frame after pressed and ends with isDown
	static bool end[256];//starts a frame after released and ends with start of isDown


	static long int mouseX, mouseY;
	static long int changeX, changeY;
	static long int lockX, lockY;
	static bool lock;

	static double deltaTime;

	static void update();

	static unsigned long long millis();
	static unsigned long long micros();
	static void hideCursor();
	static void showCursor();
	
	static unsigned char asyncGetch(unsigned int minWaitTimeMS = 100);
};
