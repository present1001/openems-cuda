/*
 * Copyright (C) 2026 openEMS contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "operator_cuda.h"
#include "engine_cuda.h"

#include <iostream>

Operator_CUDA* Operator_CUDA::New(int device, bool debugSync)
{
	std::cout << "Create FDTD operator (CUDA)" << std::endl;
	Operator_CUDA* op = new Operator_CUDA(device, debugSync);
	op->Init();
	return op;
}

Operator_CUDA::Operator_CUDA(int device, bool debugSync) :
	Operator(), m_Device(device), m_DebugSync(debugSync)
{
}

Engine* Operator_CUDA::CreateEngine()
{
	m_Engine = Engine_CUDA::New(this, m_Device, m_DebugSync);
	return m_Engine;
}
