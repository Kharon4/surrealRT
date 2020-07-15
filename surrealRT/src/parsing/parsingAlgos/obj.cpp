#include "parsing/parsingAlgos/obj.h"

#include<vector>
#include<fstream>
#include<iostream>

using namespace std;

commonMemory<meshShaded> loadModel(std::string fileNameWithExtension,chromaticShader* shader, loadAxisExchange vertexAxis) {
	ifstream file(fileNameWithExtension.c_str(), ios::in);
	vector<vec3f> vertices;
	vector<meshShaded> mesh;
	while (!file.eof()) {
		string line;
		file >> line;

		if(line == "v"){
			vec3f vertex;
			switch (vertexAxis)
			{
			case loadAxisExchange::xyz:
				file >> vertex.x >> vertex.y >> vertex.z;
				break;
			case loadAxisExchange::xzy:
				file >> vertex.x >> vertex.z >> vertex.y;
				break;
			case loadAxisExchange::yxz:
				file >> vertex.y >> vertex.x >> vertex.z;
				break;
			case loadAxisExchange::yzx:
				file >> vertex.y >> vertex.z >> vertex.x;
				break;
			case loadAxisExchange::zxy:
				file >> vertex.z >> vertex.x >> vertex.y;
				break;
			case loadAxisExchange::zyx:
				file >> vertex.z >> vertex.y >> vertex.x;
				break;
			default:
				//throw error
				break;
			}
			vertices.push_back(vertex);
		}
		else if (line == "f") {
			size_t id;
			char c;
			meshShaded ms;

			file >> id >> c;
			ms.M.pts[0] = vertices[id - 1];
			file >> id;
			file.clear();
			file >> c;
			file >> id;

			file >> id >> c;
			ms.M.pts[1] = vertices[id - 1];
			file >> id;
			file.clear();
			file >> c;
			file >> id;

			file >> id >> c;
			ms.M.pts[2] = vertices[id - 1];
			file >> id;
			file.clear();
			file >> c;
			file >> id;
			ms.colShader = shader;
			mesh.push_back(ms);
		}
	}

	commonMemory<meshShaded> rVal(mesh.size());
	for (size_t i = 0; i < mesh.size(); ++i) {
		rVal.getHost()[i] = mesh[i];
	}
	return rVal;
}


//host only function
//returns no meshShaded loaded , if -ve , function failed , the -ve val is baffMaxSize - Actaul size
long int loadModel(meshShaded* OUTBuff, unsigned int buffMaxSize, std::string fileNameWithExtension, chromaticShader* shader, loadAxisExchange vertexAxis){
	ifstream file(fileNameWithExtension.c_str(), ios::in);
	vector<vec3f> vertices;
	vector<meshShaded> mesh;
	while (!file.eof()) {
		string line;
		file >> line;

		if (line == "v") {
			vec3f vertex;
			switch (vertexAxis)
			{
			case loadAxisExchange::xyz:
				file >> vertex.x >> vertex.y >> vertex.z;
				break;
			case loadAxisExchange::xzy:
				file >> vertex.x >> vertex.z >> vertex.y;
				break;
			case loadAxisExchange::yxz:
				file >> vertex.y >> vertex.x >> vertex.z;
				break;
			case loadAxisExchange::yzx:
				file >> vertex.y >> vertex.z >> vertex.x;
				break;
			case loadAxisExchange::zxy:
				file >> vertex.z >> vertex.x >> vertex.y;
				break;
			case loadAxisExchange::zyx:
				file >> vertex.z >> vertex.y >> vertex.x;
				break;
			default:
				//throw error
				break;
			}
			vertices.push_back(vertex);
		}
		else if (line == "f") {
			size_t id;
			char c;
			meshShaded ms;

			file >> id >> c;
			ms.M.pts[0] = vertices[id - 1];
			file >> id;
			file.clear();
			file >> c;
			file >> id;

			file >> id >> c;
			ms.M.pts[1] = vertices[id - 1];
			file >> id;
			file.clear();
			file >> c;
			file >> id;

			file >> id >> c;
			ms.M.pts[2] = vertices[id - 1];
			file >> id;
			file.clear();
			file >> c;
			file >> id;
			ms.colShader = shader;
			mesh.push_back(ms);
		}
	}

	long int rVal = buffMaxSize - (mesh.size());
	if (rVal < 0)return rVal;
	for (size_t i = 0; i < mesh.size(); ++i) {
		OUTBuff[i] = mesh[i];
	}
	return mesh.size();
}

//dissable shader
disableShaderCPU dissableShader;
void loadBlankModel(meshShaded* OUTBuff, unsigned int buffMaxSize) {
	for (unsigned int i = 0; i < buffMaxSize; ++i) {
		OUTBuff[i].colShader = dissableShader.getGPUPtr();
	}
}

void loadModelVertices(std::vector<vec3d>& OUTdata, std::istream& f, loadAxisExchange vertexAxis) {
	OUTdata.resize(0);
	while (!f.eof()) {
		string line;
		f >> line;

		if (line == "v") {
			vec3f vertex;
			switch (vertexAxis)
			{
			case loadAxisExchange::xyz:
				f >> vertex.x >> vertex.y >> vertex.z;
				break;
			case loadAxisExchange::xzy:
				f >> vertex.x >> vertex.z >> vertex.y;
				break;
			case loadAxisExchange::yxz:
				f >> vertex.y >> vertex.x >> vertex.z;
				break;
			case loadAxisExchange::yzx:
				f >> vertex.y >> vertex.z >> vertex.x;
				break;
			case loadAxisExchange::zxy:
				f >> vertex.z >> vertex.x >> vertex.y;
				break;
			case loadAxisExchange::zyx:
				f >> vertex.z >> vertex.y >> vertex.x;
				break;
			default:
				//throw error
				break;
			}
			OUTdata.push_back(vertex);
		}
	}
}