#pragma once

#include "mesh.h"
#include "camera.h"
#include "light.h"

#include <vector>


typedef unsigned char BYTE;

collTriangle* initMesh(triangle* list, size_t noOfTrinagles);

using std::vector;
void render(camera c, triangle* trs, collTriangle* collTrs, long long noTrs,const vector<pointLight> &pLights ,const vector<directionalLight> &dLights, BYTE* dataOut);