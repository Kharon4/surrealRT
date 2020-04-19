#pragma once
#include "rayTrace.h"
#include <math.h>

#define noHalfLongitudes 10
#define noPtsPerHalfLongitude 5 //other than the ends
#define noTrs noHalfLongitudes

triangle* generateTriangles() {
	triangle* rval = new triangle[noTrs];
	
	return rval;
}


void deleteTrs(triangle* trs, collTriangle* cTrs, long long no = noTrs) {
	delete[] cTrs;
	for (long long i = 0; i < no; ++i) {
		delete trs[i].pts[0];
		delete trs[i].pts[1];
		delete trs[i].pts[2];
	}
	delete[] trs;
}