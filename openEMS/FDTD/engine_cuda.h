/*
 * Copyright (C) 2026 openEMS contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include "engine.h"

class Engine_CUDA : public Engine
{
public:
	static Engine_CUDA* New(const Operator* op, int device, bool debugSync);
	virtual ~Engine_CUDA();

	virtual void Init();
	virtual void Reset();
	virtual bool IterateTS(unsigned int iterTS);
	virtual void PrepareProcessing();

	int GetCUDADevice() const {return m_Device;}

protected:
	Engine_CUDA(const Operator* op, int device, bool debugSync);

private:
	struct Impl;
	Impl* m_Impl;
	int m_Device;
	bool m_DebugSync;
	bool m_Initialized;

	void ValidateAndUploadExtensions();
	void UploadExcitation(class Operator_Ext_Excitation* excitation);
	void UploadTFSF(class Operator_Ext_TFSF* tfsf);
	void UploadSteadyState(class Operator_Ext_SteadyState* steadyState);
	void UpdateSteadyState(unsigned int threads);
	void UploadUPML(class Operator_Ext_UPML* upml);
	void UploadLumpedRLC(class Operator_Ext_LumpedRLC* lumpedRLC);
};

#endif // ENGINE_CUDA_H
