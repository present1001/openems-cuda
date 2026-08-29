/*
 * Copyright (C) 2026 openEMS contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OPERATOR_CUDA_H
#define OPERATOR_CUDA_H

#include "operator.h"

class Operator_CUDA : public Operator
{
public:
	static Operator_CUDA* New(int device, bool debugSync);
	virtual ~Operator_CUDA() {}

	virtual Engine* CreateEngine();

	int GetCUDADevice() const {return m_Device;}
	bool GetCUDADebugSync() const {return m_DebugSync;}

protected:
	Operator_CUDA(int device, bool debugSync);

private:
	int m_Device;
	bool m_DebugSync;
};

#endif // OPERATOR_CUDA_H
