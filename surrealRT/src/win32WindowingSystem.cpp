#include "win32WindowingSystem.h"

#include <iostream>

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

