#pragma once
enum commonMemType { hostOnly, deviceOnly, both };

template <typename T>
class commonMemory {
private:
	void* hostPtr = nullptr;
	void* devicePtr = nullptr;
	bool hostUpdated = true;
	size_t noElements;
	size_t size;
	commonMemType type;
public:

	__host__
	commonMemory(size_t Size = 1, commonMemType Type = commonMemType::both);

	size_t getNoElements();

	T* getHost(bool* OUTupdated = nullptr);//OUTupdated = true when data is copied
	T* getDevice(bool* OUTupdated = nullptr);//OUTupdated = true when data is copied

	__host__
	~commonMemory();
};

#ifndef _commonMemory_declarations_only
#ifndef _INSIDE_commonMemory_Header
#include "cudaRelated/commonMemory.cu"
#endif
#endif // !_commonMemory_declarations_only
