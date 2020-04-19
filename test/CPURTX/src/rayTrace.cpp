#include "rayTrace.h"

#define defaultColor vec3f(100, 0, 100)

collTriangle* initMesh(triangle* list, size_t noOfTrinagles) {
	collTriangle* rVal = nullptr;
	rVal = new collTriangle[noOfTrinagles];
	for (size_t i = 0; i < noOfTrinagles; ++i) {
		rVal[i].calc(list[i]);
	}
	return rVal;
}

long long int getClosestIntersection(linearMathD::line ray, collTriangle* collTrs,long long noOfTrs,vec3d * minCollPt = nullptr, double * minDist = nullptr) {
	long long closestId = -1;
	double smallestDist = -1;
	vec3d collPt;
	double tempDist;
	for (long long i = 0; i < noOfTrs; ++i) {
		collPt = linearMathD::rayCast(ray, collTrs[i].collPlane);
		if (linearMathD::errCode)continue;
		//check if inside
		if (vec3d::dot(collTrs[i].sidePlanes[0].getDr(), collPt - collTrs[i].sidePlanes[0].getPt()) < 0)continue;
		if (vec3d::dot(collTrs[i].sidePlanes[1].getDr(), collPt - collTrs[i].sidePlanes[1].getPt()) < 0)continue;
		if (vec3d::dot(collTrs[i].sidePlanes[2].getDr(), collPt - collTrs[i].sidePlanes[2].getPt()) < 0)continue;

		tempDist = vec3d::dot(ray.getDr(), collPt - ray.getPt());
		if (tempDist < smallestDist || smallestDist < 0) {
			smallestDist = tempDist;
			closestId = i;
			if (minCollPt != nullptr)*minCollPt = collPt;
		}
	}
	if (minDist != nullptr)*minDist = smallestDist/ray.getDr().mag();
	return closestId;
}

vec3f rayTrace(linearMathD::line ray, triangle* trs, collTriangle* collTrs, long long noTrs,const vector<pointLight>& pLights, const vector<directionalLight>& dLights,unsigned char iterations = 3) {
	if (iterations == 0)return defaultColor;
	vec3d collPt;
	long long id = getClosestIntersection(ray, collTrs, noTrs,&collPt);
	if (id < 0)return defaultColor;
	vec3f finalColor = vec3f(0, 0, 0);

	//global illuminantion
	finalColor += trs[id].diffuseRefelctivity;
	
	//point light


	//reflectance
	{
		linearMathD::line reflectedRay;
		reflectedRay.setPT(collPt - ray.getDr()*0.01);
		reflectedRay.setDR(linearMathD::getMirrorImage(collPt + ray.getDr(), collTrs[id].collPlane) - collPt);
		vec3f reflectedColor = rayTrace(reflectedRay,trs,collTrs,noTrs,pLights,dLights,iterations-1);
		reflectedColor.x *= trs[id].reflectivity.x;
		reflectedColor.y *= trs[id].reflectivity.y;
		reflectedColor.z *= trs[id].reflectivity.z;
		finalColor += reflectedColor;
	}
	return finalColor;
}

void convertToByteColor(float min, float max, vec3f color, BYTE* OUTColor) {
	color -= vec3f(min, min, min);
	color *= 256 / (max - min);
	if (color.x < 0)color.x = 0;
	if (color.y < 0)color.y = 0;
	if (color.z < 0)color.z = 0;
	if (color.x > 255)color.x = 255;
	if (color.y > 255)color.y = 255;
	if (color.z > 255)color.z = 255;
	OUTColor[2] = color.x;//r
	OUTColor[1] = color.y;//g
	OUTColor[0] = color.z;//b
}

void render(camera c, triangle* trs, collTriangle* collTrs, long long noTrs, const vector<pointLight>& pLights, const vector<directionalLight>& dLights, BYTE* dataOut){
	int pixelId = 0;
	for (short y = 0; y < c.yRes; ++y) {
		for (short x = 0; x < c.xRes; ++x , pixelId+=3) {
			vec3f col = rayTrace(getRay(c, x, y), trs, collTrs, noTrs,pLights,dLights);
			convertToByteColor(0, 256, col, dataOut + pixelId);
		}
	}
}