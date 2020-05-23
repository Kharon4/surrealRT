#include "parsing/parsingAlgos/obj.h"

#include<vector>
#include<fstream>


using namespace std;

commonMemory<meshShaded> loadModel(std::string fileNameWithExtension,chromaticShader* shader) {
	ifstream file(fileNameWithExtension.c_str(), ios::in);
	vector<vec3d> vertices;
	vector<meshShaded> mesh;
	while (!file.eof()) {
		string line;
		file >> line;

		if(line == "v"){
			vec3d vertex;
			file >> vertex.x >> vertex.y >> vertex.z;
			vertices.push_back(vertex);
		}
		else if (line == "f") {
			size_t id;
			char c;
			meshShaded ms;

			file >> id >> c;
			ms.M.pts[0] = vertices[id - 1];
			file >> id >> c;
			file >> id;

			file >> id >> c;
			ms.M.pts[1] = vertices[id - 1];
			file >> id >> c;
			file >> id;

			file >> id >> c;
			ms.M.pts[2] = vertices[id - 1];
			file >> id >> c;
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