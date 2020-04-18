#include "camera.h"

linearMathD::line getRay(camera c,short x, short y) {
	linearMathD::line rVal;
	rVal.setPT(c.vertex);
	rVal.setDR(c.topLeftCorner + c.down * ((y + 0.5) / c.yRes) + c.right * ((x + 0.5) / c.xRes) - c.vertex);
	return rVal;
}