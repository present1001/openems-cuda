/*
 * Copyright (C) 2026 openEMS contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "engine_cuda.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/operator_ext_steadystate.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/operator_extension.h"
#include "extensions/engine_ext_steadystate.h"
#include "excitation.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{

void CudaCheck(cudaError_t status, const char* operation)
{
	if (status == cudaSuccess)
		return;
	std::ostringstream message;
	message << "CUDA " << operation << " failed: "
		<< cudaGetErrorName(status) << " (" << cudaGetErrorString(status) << ")";
	throw std::runtime_error(message.str());
}

template <typename T>
void DeviceAllocate(T*& pointer, size_t count)
{
	if (count == 0)
	{
		pointer = NULL;
		return;
	}
	CudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)),
		"memory allocation");
}

template <typename T>
void DeviceAllocateAndCopy(T*& pointer, const T* source, size_t count)
{
	DeviceAllocate(pointer, count);
	if (count != 0)
		CudaCheck(cudaMemcpy(pointer, source, count * sizeof(T),
			cudaMemcpyHostToDevice), "host-to-device copy");
}

template <typename T>
void DeviceFree(T*& pointer)
{
	if (pointer != NULL)
		cudaFree(pointer);
	pointer = NULL;
}

__device__ __forceinline__ size_t FieldIndex(
	unsigned int component,
	unsigned int x,
	unsigned int y,
	unsigned int z,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz)
{
	const size_t cells = static_cast<size_t>(nx) * ny * nz;
	return static_cast<size_t>(component) * cells
		+ (static_cast<size_t>(x) * ny + y) * nz + z;
}

__global__ void UpdateVoltagesKernel(
	FDTD_FLOAT* volt,
	const FDTD_FLOAT* curr,
	const FDTD_FLOAT* vv,
	const FDTD_FLOAT* vi,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	size_t cells)
{
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x
		+ threadIdx.x;
	if (linear >= cells)
		return;

	const unsigned int z = static_cast<unsigned int>(linear % nz);
	const size_t xy = linear / nz;
	const unsigned int y = static_cast<unsigned int>(xy % ny);
	const unsigned int x = static_cast<unsigned int>(xy / ny);
	const unsigned int xm = x == 0 ? 0 : x - 1;
	const unsigned int ym = y == 0 ? 0 : y - 1;
	const unsigned int zm = z == 0 ? 0 : z - 1;

	const size_t v0 = FieldIndex(0, x, y, z, nx, ny, nz);
	const size_t v1 = FieldIndex(1, x, y, z, nx, ny, nz);
	const size_t v2 = FieldIndex(2, x, y, z, nx, ny, nz);

	FDTD_FLOAT updated = volt[v0] * vv[v0];
	updated += vi[v0] * (
		curr[FieldIndex(2, x, y,  z, nx, ny, nz)]
		- curr[FieldIndex(2, x, ym, z, nx, ny, nz)]
		- curr[FieldIndex(1, x, y,  z, nx, ny, nz)]
		+ curr[FieldIndex(1, x, y, zm, nx, ny, nz)]);
	volt[v0] = updated;

	updated = volt[v1] * vv[v1];
	updated += vi[v1] * (
		curr[FieldIndex(0, x,  y, z, nx, ny, nz)]
		- curr[FieldIndex(0, x,  y, zm, nx, ny, nz)]
		- curr[FieldIndex(2, x,  y, z, nx, ny, nz)]
		+ curr[FieldIndex(2, xm, y, z, nx, ny, nz)]);
	volt[v1] = updated;

	updated = volt[v2] * vv[v2];
	updated += vi[v2] * (
		curr[FieldIndex(1, x,  y,  z, nx, ny, nz)]
		- curr[FieldIndex(1, xm, y,  z, nx, ny, nz)]
		- curr[FieldIndex(0, x,  y,  z, nx, ny, nz)]
		+ curr[FieldIndex(0, x,  ym, z, nx, ny, nz)]);
	volt[v2] = updated;
}

__global__ void UpdateCurrentsKernel(
	FDTD_FLOAT* curr,
	const FDTD_FLOAT* volt,
	const FDTD_FLOAT* ii,
	const FDTD_FLOAT* iv,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	size_t interiorCells)
{
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x
		+ threadIdx.x;
	if (linear >= interiorCells)
		return;

	const unsigned int interiorNz = nz - 1;
	const unsigned int interiorNy = ny - 1;
	const unsigned int z = static_cast<unsigned int>(linear % interiorNz);
	const size_t xy = linear / interiorNz;
	const unsigned int y = static_cast<unsigned int>(xy % interiorNy);
	const unsigned int x = static_cast<unsigned int>(xy / interiorNy);

	const size_t c0 = FieldIndex(0, x, y, z, nx, ny, nz);
	const size_t c1 = FieldIndex(1, x, y, z, nx, ny, nz);
	const size_t c2 = FieldIndex(2, x, y, z, nx, ny, nz);

	FDTD_FLOAT updated = curr[c0] * ii[c0];
	updated += iv[c0] * (
		volt[FieldIndex(2, x, y,     z, nx, ny, nz)]
		- volt[FieldIndex(2, x, y + 1, z, nx, ny, nz)]
		- volt[FieldIndex(1, x, y,     z, nx, ny, nz)]
		+ volt[FieldIndex(1, x, y, z + 1, nx, ny, nz)]);
	curr[c0] = updated;

	updated = curr[c1] * ii[c1];
	updated += iv[c1] * (
		volt[FieldIndex(0, x,     y, z, nx, ny, nz)]
		- volt[FieldIndex(0, x,     y, z + 1, nx, ny, nz)]
		- volt[FieldIndex(2, x,     y, z, nx, ny, nz)]
		+ volt[FieldIndex(2, x + 1, y, z, nx, ny, nz)]);
	curr[c1] = updated;

	updated = curr[c2] * ii[c2];
	updated += iv[c2] * (
		volt[FieldIndex(1, x,     y,     z, nx, ny, nz)]
		- volt[FieldIndex(1, x + 1, y,     z, nx, ny, nz)]
		- volt[FieldIndex(0, x,     y,     z, nx, ny, nz)]
		+ volt[FieldIndex(0, x,     y + 1, z, nx, ny, nz)]);
	curr[c2] = updated;
}

__global__ void ApplyExcitationKernel(
	FDTD_FLOAT* field,
	const size_t* indices,
	const FDTD_FLOAT* amplitudes,
	const unsigned int* delays,
	unsigned int count,
	const FDTD_FLOAT* signal,
	unsigned int signalLength,
	unsigned int periodSteps,
	unsigned int timestep)
{
	const unsigned int entry = blockIdx.x * blockDim.x + threadIdx.x;
	if (entry >= count)
		return;

	int signalPosition = static_cast<int>(timestep)
		- static_cast<int>(delays[entry]);
	signalPosition *= signalPosition > 0;
	const int period = periodSteps == 0
		? static_cast<int>(timestep + 1) : static_cast<int>(periodSteps);
	signalPosition %= period;
	signalPosition *= signalPosition < static_cast<int>(signalLength);
	field[indices[entry]] += amplitudes[entry] * signal[signalPosition];
}

__device__ __forceinline__ FDTD_FLOAT DelayedSignalValue(
	const FDTD_FLOAT* signal,
	unsigned int signalLength,
	unsigned int periodSteps,
	unsigned int timestep,
	unsigned int delay)
{
	unsigned int signalPosition = 0;
	if (timestep >= delay)
	{
		signalPosition = timestep - delay;
		if (periodSteps != 0)
			signalPosition %= periodSteps;
		else if (signalPosition >= signalLength)
			signalPosition = 0;
	}
	return signal[signalPosition];
}

__global__ void ApplyTFSFCorrectionKernel(
	FDTD_FLOAT* field,
	const size_t* indices,
	const FDTD_FLOAT* amplitudes,
	const unsigned int* delays,
	const FDTD_FLOAT* delayDeltas,
	unsigned int count,
	const FDTD_FLOAT* signal,
	unsigned int signalLength,
	unsigned int periodSteps,
	unsigned int timestep)
{
	const unsigned int entry = blockIdx.x * blockDim.x + threadIdx.x;
	if (entry >= count)
		return;

	const unsigned int delay = delays[entry];
	const FDTD_FLOAT delta = delayDeltas[entry];
	const FDTD_FLOAT value = amplitudes[entry] * (
		(1.0f - delta) * DelayedSignalValue(signal, signalLength,
			periodSteps, timestep, delay)
		+ delta * DelayedSignalValue(signal, signalLength,
			periodSteps, timestep, delay + 1));
	atomicAdd(field + indices[entry], value);
}

__global__ void RecordSteadyStateProbesKernel(
	const FDTD_FLOAT* field,
	const size_t* indices,
	FDTD_FLOAT* records,
	unsigned int probeCount,
	unsigned int recordLength,
	unsigned int recordPosition)
{
	const unsigned int probe = blockIdx.x * blockDim.x + threadIdx.x;
	if (probe >= probeCount)
		return;
	records[static_cast<size_t>(probe) * recordLength + recordPosition]
		= field[indices[probe]];
}

__global__ void SteadyStateEnergyKernel(
	const FDTD_FLOAT* voltage,
	const FDTD_FLOAT* current,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	size_t interiorCells,
	double* energy)
{
	extern __shared__ double partial[];
	const unsigned int thread = threadIdx.x;
	const size_t interiorNz = nz - 1;
	const size_t interiorNy = ny - 1;
	double sum = 0.0;
	for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + thread;
		linear < interiorCells;
		linear += static_cast<size_t>(gridDim.x) * blockDim.x)
	{
		const unsigned int z = static_cast<unsigned int>(linear % interiorNz);
		const size_t xy = linear / interiorNz;
		const unsigned int y = static_cast<unsigned int>(xy % interiorNy);
		const unsigned int x = static_cast<unsigned int>(xy / interiorNy);
		for (unsigned int component = 0; component < 3; ++component)
		{
			const size_t index = FieldIndex(component, x, y, z, nx, ny, nz);
			const double e = voltage[index];
			const double h = current[index];
			sum += e * e + h * h;
		}
	}
	partial[thread] = sum;
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1)
	{
		if (thread < stride)
			partial[thread] += partial[thread + stride];
		__syncthreads();
	}
	if (thread == 0)
		atomicAdd(energy, partial[0]);
}

__global__ void UPMLPreKernel(
	FDTD_FLOAT* field,
	FDTD_FLOAT* flux,
	const FDTD_FLOAT* retain,
	const FDTD_FLOAT* oldFlux,
	unsigned int startX,
	unsigned int startY,
	unsigned int startZ,
	unsigned int regionNx,
	unsigned int regionNy,
	unsigned int regionNz,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	size_t regionCells)
{
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x
		+ threadIdx.x;
	if (linear >= regionCells)
		return;

	const unsigned int localZ = static_cast<unsigned int>(linear % regionNz);
	const size_t localXY = linear / regionNz;
	const unsigned int localY = static_cast<unsigned int>(localXY % regionNy);
	const unsigned int localX = static_cast<unsigned int>(localXY / regionNy);
	const unsigned int x = startX + localX;
	const unsigned int y = startY + localY;
	const unsigned int z = startZ + localZ;

	for (unsigned int component = 0; component < 3; ++component)
	{
		const size_t localIndex = static_cast<size_t>(component) * regionCells
			+ linear;
		const size_t globalIndex = FieldIndex(component, x, y, z, nx, ny, nz);
		const FDTD_FLOAT nextFlux = retain[localIndex] * field[globalIndex]
			- oldFlux[localIndex] * flux[localIndex];
		field[globalIndex] = flux[localIndex];
		flux[localIndex] = nextFlux;
	}
}

__global__ void UPMLPostKernel(
	FDTD_FLOAT* field,
	FDTD_FLOAT* flux,
	const FDTD_FLOAT* newFlux,
	unsigned int startX,
	unsigned int startY,
	unsigned int startZ,
	unsigned int regionNx,
	unsigned int regionNy,
	unsigned int regionNz,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	size_t regionCells)
{
	const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x
		+ threadIdx.x;
	if (linear >= regionCells)
		return;

	const unsigned int localZ = static_cast<unsigned int>(linear % regionNz);
	const size_t localXY = linear / regionNz;
	const unsigned int localY = static_cast<unsigned int>(localXY % regionNy);
	const unsigned int localX = static_cast<unsigned int>(localXY / regionNy);
	const unsigned int x = startX + localX;
	const unsigned int y = startY + localY;
	const unsigned int z = startZ + localZ;

	for (unsigned int component = 0; component < 3; ++component)
	{
		const size_t localIndex = static_cast<size_t>(component) * regionCells
			+ linear;
		const size_t globalIndex = FieldIndex(component, x, y, z, nx, ny, nz);
		const FDTD_FLOAT previousFlux = flux[localIndex];
		flux[localIndex] = field[globalIndex];
		field[globalIndex] = previousFlux
			+ newFlux[localIndex] * flux[localIndex];
	}
}

__global__ void LumpedRLCPreVoltageKernel(
	double* inductorCurrent,
	const double* previousVoltage,
	const double* currentToVoltage,
	const double* inductorToVoltage,
	unsigned int count)
{
	const unsigned int entry = blockIdx.x * blockDim.x + threadIdx.x;
	if (entry >= count)
		return;
	inductorCurrent[entry] += currentToVoltage[entry]
		* inductorToVoltage[entry] * previousVoltage[entry];
}

__global__ void LumpedRLCApplyVoltageKernel(
	FDTD_FLOAT* field,
	const size_t* indices,
	double* inductorCurrent,
	double* voltage0,
	const double* voltage1,
	const double* voltage2,
	double* current0,
	const double* current1,
	const double* current2,
	const double* vv2,
	const double* vj1,
	const double* vj2,
	const double* vvd,
	const double* ib0,
	const double* b1,
	const double* b2,
	unsigned int count)
{
	const unsigned int entry = blockIdx.x * blockDim.x + threadIdx.x;
	if (entry >= count)
		return;

	double voltage = field[indices[entry]];
	voltage = vvd[entry] * (
		voltage - inductorCurrent[entry]
		+ vv2[entry] * voltage2[entry]
		+ vj1[entry] * current1[entry]
		+ vj2[entry] * current2[entry]);
	voltage0[entry] = voltage;
	current0[entry] = ib0[entry] * (voltage - voltage2[entry])
		- b1[entry] * ib0[entry] * current1[entry]
		- b2[entry] * ib0[entry] * current2[entry];
	field[indices[entry]] = voltage;
}

unsigned int BlockCount(size_t workItems, unsigned int threads)
{
	return static_cast<unsigned int>((workItems + threads - 1) / threads);
}

} // namespace

struct Engine_CUDA::Impl
{
	struct ExcitationSet
	{
		size_t* indices;
		FDTD_FLOAT* amplitudes;
		unsigned int* delays;
		FDTD_FLOAT* signal;
		unsigned int count;
		unsigned int signalLength;
		unsigned int periodSteps;

		ExcitationSet() : indices(NULL), amplitudes(NULL), delays(NULL),
			signal(NULL), count(0), signalLength(0), periodSteps(0) {}
	};

	struct TFSFCorrectionSet
	{
		size_t* indices;
		FDTD_FLOAT* amplitudes;
		unsigned int* delays;
		FDTD_FLOAT* delayDeltas;
		FDTD_FLOAT* signal;
		unsigned int count;
		unsigned int signalLength;
		unsigned int periodSteps;

		TFSFCorrectionSet() : indices(NULL), amplitudes(NULL), delays(NULL),
			delayDeltas(NULL), signal(NULL), count(0), signalLength(0),
			periodSteps(0) {}
	};

	struct SteadyStateData
	{
		size_t* probeIndices;
		FDTD_FLOAT* records;
		double* energy;
		Engine_Ext_SteadyState* engine;
		unsigned int probeCount;
		unsigned int periodSteps;

		SteadyStateData() : probeIndices(NULL), records(NULL), energy(NULL),
			engine(NULL), probeCount(0), periodSteps(0) {}
	};

	struct PMLRegion
	{
		unsigned int start[3];
		unsigned int extent[3];
		size_t cells;
		FDTD_FLOAT* vv;
		FDTD_FLOAT* vvfo;
		FDTD_FLOAT* vvfn;
		FDTD_FLOAT* ii;
		FDTD_FLOAT* iifo;
		FDTD_FLOAT* iifn;
		FDTD_FLOAT* voltageFlux;
		FDTD_FLOAT* currentFlux;

		PMLRegion() : cells(0), vv(NULL), vvfo(NULL), vvfn(NULL), ii(NULL),
			iifo(NULL), iifn(NULL), voltageFlux(NULL), currentFlux(NULL)
		{
			start[0] = start[1] = start[2] = 0;
			extent[0] = extent[1] = extent[2] = 0;
		}
	};

	struct LumpedRLCData
	{
		size_t* indices;
		double* ilv;
		double* i2v;
		double* vv2;
		double* vj1;
		double* vj2;
		double* vvd;
		double* ib0;
		double* b1;
		double* b2;
		double* inductorCurrent;
		double* voltage[3];
		double* current[3];
		unsigned int count;

		LumpedRLCData() : indices(NULL), ilv(NULL), i2v(NULL), vv2(NULL),
			vj1(NULL), vj2(NULL), vvd(NULL), ib0(NULL), b1(NULL), b2(NULL),
			inductorCurrent(NULL), count(0)
		{
			for (unsigned int n = 0; n < 3; ++n)
			{
				voltage[n] = NULL;
				current[n] = NULL;
			}
		}
	};

	FDTD_FLOAT* volt;
	FDTD_FLOAT* curr;
	FDTD_FLOAT* vv;
	FDTD_FLOAT* vi;
	FDTD_FLOAT* ii;
	FDTD_FLOAT* iv;
	size_t cells;
	size_t fieldElements;
	ExcitationSet voltageExcitation;
	ExcitationSet currentExcitation;
	TFSFCorrectionSet tfsfVoltage;
	TFSFCorrectionSet tfsfCurrent;
	SteadyStateData steadyState;
	std::vector<PMLRegion> pmlRegions;
	LumpedRLCData lumpedRLC;
	bool voltageHostPinned;
	bool currentHostPinned;

	Impl() : volt(NULL), curr(NULL), vv(NULL), vi(NULL), ii(NULL), iv(NULL),
		cells(0), fieldElements(0), voltageHostPinned(false),
		currentHostPinned(false) {}
};

Engine_CUDA* Engine_CUDA::New(const Operator* op, int device, bool debugSync)
{
	std::unique_ptr<Engine_CUDA> engine(new Engine_CUDA(op, device, debugSync));
	engine->Init();
	return engine.release();
}

Engine_CUDA::Engine_CUDA(const Operator* op, int device, bool debugSync) :
	Engine(op), m_Impl(new Impl()), m_Device(device), m_DebugSync(debugSync),
	m_Initialized(false)
{
	m_type = CUDA;
}

Engine_CUDA::~Engine_CUDA()
{
	Reset();
	delete m_Impl;
	m_Impl = NULL;
}

void Engine_CUDA::Init()
{
	Reset();
	try
	{
		CudaCheck(cudaSetDevice(m_Device), "device selection");

		cudaDeviceProp properties;
		CudaCheck(cudaGetDeviceProperties(&properties, m_Device),
			"device property query");

		numTS = 0;
		volt_ptr = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("volt", numLines);
		curr_ptr = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("curr", numLines);

		m_Impl->cells = static_cast<size_t>(numLines[0])
			* numLines[1] * numLines[2];
		m_Impl->fieldElements = 3 * m_Impl->cells;
		const size_t fieldBytes = m_Impl->fieldElements * sizeof(FDTD_FLOAT);
		cudaError_t pinStatus = cudaHostRegister(
			volt_ptr->data(), fieldBytes, cudaHostRegisterPortable);
		if (pinStatus == cudaSuccess)
			m_Impl->voltageHostPinned = true;
		else
		{
			std::cerr << "CUDA warning: voltage host mirror could not be pinned: "
				<< cudaGetErrorString(pinStatus) << std::endl;
			cudaGetLastError();
		}
		pinStatus = cudaHostRegister(
			curr_ptr->data(), fieldBytes, cudaHostRegisterPortable);
		if (pinStatus == cudaSuccess)
			m_Impl->currentHostPinned = true;
		else
		{
			std::cerr << "CUDA warning: current host mirror could not be pinned: "
				<< cudaGetErrorString(pinStatus) << std::endl;
			cudaGetLastError();
		}

		size_t freeBytes = 0;
		size_t totalBytes = 0;
		CudaCheck(cudaMemGetInfo(&freeBytes, &totalBytes), "memory query");
		const size_t baseRequiredBytes = 6 * fieldBytes;
		if (baseRequiredBytes > freeBytes)
		{
			std::ostringstream message;
			message << "CUDA engine needs at least "
				<< (baseRequiredBytes / (1024 * 1024))
				<< " MiB for fields and coefficients, but device "
				<< m_Device << " has only " << (freeBytes / (1024 * 1024))
				<< " MiB free";
			throw std::runtime_error(message.str());
		}

		DeviceAllocate(m_Impl->volt, m_Impl->fieldElements);
		DeviceAllocate(m_Impl->curr, m_Impl->fieldElements);
		CudaCheck(cudaMemset(m_Impl->volt, 0, fieldBytes), "voltage initialization");
		CudaCheck(cudaMemset(m_Impl->curr, 0, fieldBytes), "current initialization");
		DeviceAllocateAndCopy(m_Impl->vv, Op->vv_ptr->data(), m_Impl->fieldElements);
		DeviceAllocateAndCopy(m_Impl->vi, Op->vi_ptr->data(), m_Impl->fieldElements);
		DeviceAllocateAndCopy(m_Impl->ii, Op->ii_ptr->data(), m_Impl->fieldElements);
		DeviceAllocateAndCopy(m_Impl->iv, Op->iv_ptr->data(), m_Impl->fieldElements);

		ValidateAndUploadExtensions();
		m_Initialized = true;

		std::cout << "Create CUDA FDTD engine on device " << m_Device
			<< ": " << properties.name << " (compute capability "
			<< properties.major << "." << properties.minor << ")" << std::endl;
		std::cout << "CUDA base storage: "
			<< (static_cast<double>(baseRequiredBytes) / (1024.0 * 1024.0))
			<< " MiB for " << m_Impl->cells << " Yee cells" << std::endl;
	}
	catch (...)
	{
		Reset();
		throw;
	}
}

void Engine_CUDA::Reset()
{
	if (m_Impl != NULL)
	{
		if (m_Initialized)
			cudaSetDevice(m_Device);

		DeviceFree(m_Impl->voltageExcitation.indices);
		DeviceFree(m_Impl->voltageExcitation.amplitudes);
		DeviceFree(m_Impl->voltageExcitation.delays);
		DeviceFree(m_Impl->voltageExcitation.signal);
		m_Impl->voltageExcitation = Impl::ExcitationSet();

		DeviceFree(m_Impl->currentExcitation.indices);
		DeviceFree(m_Impl->currentExcitation.amplitudes);
		DeviceFree(m_Impl->currentExcitation.delays);
		DeviceFree(m_Impl->currentExcitation.signal);
		m_Impl->currentExcitation = Impl::ExcitationSet();

		DeviceFree(m_Impl->tfsfVoltage.indices);
		DeviceFree(m_Impl->tfsfVoltage.amplitudes);
		DeviceFree(m_Impl->tfsfVoltage.delays);
		DeviceFree(m_Impl->tfsfVoltage.delayDeltas);
		DeviceFree(m_Impl->tfsfVoltage.signal);
		m_Impl->tfsfVoltage = Impl::TFSFCorrectionSet();

		DeviceFree(m_Impl->tfsfCurrent.indices);
		DeviceFree(m_Impl->tfsfCurrent.amplitudes);
		DeviceFree(m_Impl->tfsfCurrent.delays);
		DeviceFree(m_Impl->tfsfCurrent.delayDeltas);
		DeviceFree(m_Impl->tfsfCurrent.signal);
		m_Impl->tfsfCurrent = Impl::TFSFCorrectionSet();

		DeviceFree(m_Impl->steadyState.probeIndices);
		DeviceFree(m_Impl->steadyState.records);
		DeviceFree(m_Impl->steadyState.energy);
		m_Impl->steadyState = Impl::SteadyStateData();

		for (size_t n = 0; n < m_Impl->pmlRegions.size(); ++n)
		{
			Impl::PMLRegion& region = m_Impl->pmlRegions[n];
			DeviceFree(region.vv);
			DeviceFree(region.vvfo);
			DeviceFree(region.vvfn);
			DeviceFree(region.ii);
			DeviceFree(region.iifo);
			DeviceFree(region.iifn);
			DeviceFree(region.voltageFlux);
			DeviceFree(region.currentFlux);
		}
		m_Impl->pmlRegions.clear();

		DeviceFree(m_Impl->lumpedRLC.indices);
		DeviceFree(m_Impl->lumpedRLC.ilv);
		DeviceFree(m_Impl->lumpedRLC.i2v);
		DeviceFree(m_Impl->lumpedRLC.vv2);
		DeviceFree(m_Impl->lumpedRLC.vj1);
		DeviceFree(m_Impl->lumpedRLC.vj2);
		DeviceFree(m_Impl->lumpedRLC.vvd);
		DeviceFree(m_Impl->lumpedRLC.ib0);
		DeviceFree(m_Impl->lumpedRLC.b1);
		DeviceFree(m_Impl->lumpedRLC.b2);
		DeviceFree(m_Impl->lumpedRLC.inductorCurrent);
		for (unsigned int n = 0; n < 3; ++n)
		{
			DeviceFree(m_Impl->lumpedRLC.voltage[n]);
			DeviceFree(m_Impl->lumpedRLC.current[n]);
		}
		m_Impl->lumpedRLC = Impl::LumpedRLCData();

		DeviceFree(m_Impl->volt);
		DeviceFree(m_Impl->curr);
		DeviceFree(m_Impl->vv);
		DeviceFree(m_Impl->vi);
		DeviceFree(m_Impl->ii);
		DeviceFree(m_Impl->iv);
		if (m_Impl->voltageHostPinned && volt_ptr != NULL)
			cudaHostUnregister(volt_ptr->data());
		if (m_Impl->currentHostPinned && curr_ptr != NULL)
			cudaHostUnregister(curr_ptr->data());
		m_Impl->voltageHostPinned = false;
		m_Impl->currentHostPinned = false;
		m_Impl->cells = 0;
		m_Impl->fieldElements = 0;
	}
	m_Initialized = false;
	Engine::Reset();
}

void Engine_CUDA::ValidateAndUploadExtensions()
{
	Operator_Ext_Excitation* excitation = NULL;
	Operator_Ext_LumpedRLC* lumpedRLC = NULL;
	Operator_Ext_TFSF* tfsf = NULL;
	Operator_Ext_SteadyState* steadyState = NULL;
	for (size_t n = 0; n < Op->GetNumberOfExtentions(); ++n)
	{
		Operator_Extension* extension = Op->GetExtension(n);
		Operator_Ext_Excitation* candidate =
			dynamic_cast<Operator_Ext_Excitation*>(extension);
		if (candidate != NULL)
		{
			if (excitation != NULL)
				throw std::runtime_error(
					"CUDA engine supports only one excitation extension");
			excitation = candidate;
			continue;
		}

		Operator_Ext_UPML* upml = dynamic_cast<Operator_Ext_UPML*>(extension);
		if (upml != NULL)
		{
			UploadUPML(upml);
			continue;
		}

		Operator_Ext_TFSF* tfsfCandidate =
			dynamic_cast<Operator_Ext_TFSF*>(extension);
		if (tfsfCandidate != NULL)
		{
			if (tfsf != NULL)
				throw std::runtime_error(
					"CUDA engine supports only one TFSF extension");
			tfsf = tfsfCandidate;
			continue;
		}

		Operator_Ext_SteadyState* steadyStateCandidate =
			dynamic_cast<Operator_Ext_SteadyState*>(extension);
		if (steadyStateCandidate != NULL)
		{
			if (steadyState != NULL)
				throw std::runtime_error(
					"CUDA engine supports only one steady-state detector");
			steadyState = steadyStateCandidate;
			continue;
		}

		Operator_Ext_LumpedRLC* rlc =
			dynamic_cast<Operator_Ext_LumpedRLC*>(extension);
		if (rlc != NULL)
		{
			if (lumpedRLC != NULL)
				throw std::runtime_error(
					"CUDA engine supports only one lumped-RLC extension");
			lumpedRLC = rlc;
			continue;
		}

		std::ostringstream message;
		message << "CUDA engine does not yet support active extension: "
			<< extension->GetExtensionName()
			<< ". Select --engine=multithreaded for this model.";
		throw std::runtime_error(message.str());
	}

	if (excitation != NULL)
		UploadExcitation(excitation);
	if (tfsf != NULL)
		UploadTFSF(tfsf);
	if (steadyState != NULL)
		UploadSteadyState(steadyState);
	if (lumpedRLC != NULL)
		UploadLumpedRLC(lumpedRLC);
}

void Engine_CUDA::UploadSteadyState(Operator_Ext_SteadyState* steadyState)
{
	Impl::SteadyStateData& data = m_Impl->steadyState;
	data.probeCount = static_cast<unsigned int>(steadyState->m_E_probe_dir.size());
	data.periodSteps = steadyState->m_TS_period;
	if (data.periodSteps == 0)
		throw std::runtime_error("CUDA steady-state detector has a zero period");

	Engine_Extension* extension = steadyState->CreateEngineExtention();
	data.engine = dynamic_cast<Engine_Ext_SteadyState*>(extension);
	if (data.engine == NULL)
	{
		delete extension;
		throw std::runtime_error("CUDA could not create steady-state detector state");
	}
	data.engine->SetEngine(this);
	m_Eng_exts.push_back(data.engine);

	std::vector<size_t> indices(data.probeCount);
	const size_t cells = m_Impl->cells;
	const unsigned int ny = numLines[1];
	const unsigned int nz = numLines[2];
	for (unsigned int n = 0; n < data.probeCount; ++n)
	{
		indices[n] = static_cast<size_t>(steadyState->m_E_probe_dir[n]) * cells
			+ (static_cast<size_t>(steadyState->m_E_probe_pos[0][n]) * ny
				+ steadyState->m_E_probe_pos[1][n]) * nz
			+ steadyState->m_E_probe_pos[2][n];
	}
	DeviceAllocateAndCopy(data.probeIndices, indices.data(), data.probeCount);
	DeviceAllocate(data.records,
		static_cast<size_t>(data.probeCount) * 2 * data.periodSteps);
	DeviceAllocate(data.energy, 1);
}

void Engine_CUDA::UpdateSteadyState(unsigned int threads)
{
	Impl::SteadyStateData& data = m_Impl->steadyState;
	if (data.engine == NULL)
		return;

	const unsigned int recordLength = 2 * data.periodSteps;
	const unsigned int recordPosition = numTS % recordLength;
	if (data.probeCount != 0)
	{
		RecordSteadyStateProbesKernel<<<
			BlockCount(data.probeCount, threads), threads>>>(
				m_Impl->volt, data.probeIndices, data.records,
				data.probeCount, recordLength, recordPosition);
		CudaCheck(cudaGetLastError(), "steady-state probe kernel launch");
	}
	if ((numTS % data.periodSteps != 0) || (numTS < recordLength))
		return;

	CudaCheck(cudaMemset(data.energy, 0, sizeof(double)),
		"steady-state energy initialization");
	const size_t interiorCells = static_cast<size_t>(numLines[0] - 1)
		* (numLines[1] - 1) * (numLines[2] - 1);
	const unsigned int energyBlocks = std::min<unsigned int>(
		BlockCount(interiorCells, threads), 4096);
	SteadyStateEnergyKernel<<<energyBlocks, threads, threads * sizeof(double)>>>(
		m_Impl->volt, m_Impl->curr, numLines[0], numLines[1], numLines[2],
		interiorCells, data.energy);
	CudaCheck(cudaGetLastError(), "steady-state energy kernel launch");

	double currentEnergy = 0.0;
	CudaCheck(cudaMemcpy(&currentEnergy, data.energy, sizeof(double),
		cudaMemcpyDeviceToHost), "steady-state energy copy");
	std::vector<FDTD_FLOAT> records(
		static_cast<size_t>(data.probeCount) * recordLength);
	if (data.probeCount != 0)
		CudaCheck(cudaMemcpy(records.data(), data.records,
			records.size() * sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost),
			"steady-state probe copy");

	bool noValid = true;
	data.engine->m_last_max_diff = 0.0;
	if (data.engine->last_total_energy > 0.0)
	{
		data.engine->m_last_max_diff = std::fabs(
			currentEnergy - data.engine->last_total_energy)
			/ data.engine->last_total_energy;
		noValid = false;
	}
	data.engine->last_total_energy = currentEnergy;

	unsigned int oldPosition = 0;
	unsigned int newPosition = data.periodSteps;
	if (recordPosition <= data.periodSteps)
	{
		newPosition = 0;
		oldPosition = data.periodSteps;
	}
	std::vector<double> currentPower(data.probeCount, 0.0);
	std::vector<double> differencePower(data.probeCount, 0.0);
	double maximumPower = 0.0;
	for (unsigned int probe = 0; probe < data.probeCount; ++probe)
	{
		const FDTD_FLOAT* buffer = records.data()
			+ static_cast<size_t>(probe) * recordLength;
		for (unsigned int step = 0; step < data.periodSteps; ++step)
		{
			const double current = buffer[step + newPosition];
			const double difference = buffer[step + oldPosition] - current;
			currentPower[probe] += current * current;
			differencePower[probe] += difference * difference;
		}
		maximumPower = std::max(maximumPower, currentPower[probe]);
	}
	for (unsigned int probe = 0; probe < data.probeCount; ++probe)
	{
		if (currentPower[probe] > maximumPower * 1e-2)
		{
			data.engine->m_last_max_diff = std::max(
				data.engine->m_last_max_diff,
				differencePower[probe] / currentPower[probe]);
			noValid = false;
		}
	}
	if (noValid || data.engine->m_last_max_diff > 1.0)
		data.engine->m_last_max_diff = 1.0;
}

void Engine_CUDA::UploadTFSF(Operator_Ext_TFSF* tfsf)
{
	struct HostCorrections
	{
		std::vector<size_t> indices;
		std::vector<FDTD_FLOAT> amplitudes;
		std::vector<unsigned int> delays;
		std::vector<FDTD_FLOAT> delayDeltas;
	};

	HostCorrections voltage;
	HostCorrections current;
	const size_t cells = m_Impl->cells;
	const unsigned int ny = numLines[1];
	const unsigned int nz = numLines[2];

	auto append = [cells, ny, nz](HostCorrections& output,
		unsigned int component, const unsigned int pos[3], FDTD_FLOAT amplitude,
		unsigned int delay, FDTD_FLOAT delayDelta)
	{
		if (amplitude == 0)
			return;
		output.indices.push_back(static_cast<size_t>(component) * cells
			+ (static_cast<size_t>(pos[0]) * ny + pos[1]) * nz + pos[2]);
		output.amplitudes.push_back(amplitude);
		output.delays.push_back(delay);
		output.delayDeltas.push_back(delayDelta);
	};

	for (unsigned int direction = 0; direction < 3; ++direction)
	{
		const unsigned int nP = (direction + 1) % 3;
		const unsigned int nPP = (direction + 2) % 3;
		for (unsigned int side = 0; side < 2; ++side)
		{
			if (!tfsf->m_ActiveDir[direction][side])
				continue;
			unsigned int position[3];
			position[direction] = side == 0
				? tfsf->m_Start[direction] : tfsf->m_Stop[direction];
			unsigned int planePosition = 0;
			for (unsigned int i = 0; i < tfsf->m_numLines[nP]; ++i)
			{
				position[nP] = tfsf->m_Start[nP] + i;
				for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; ++j)
				{
					position[nPP] = tfsf->m_Start[nPP] + j;
					for (unsigned int component = 0; component < 2; ++component)
					{
						const unsigned int fieldComponent = component == 0 ? nP : nPP;
						append(voltage, fieldComponent, position,
							tfsf->m_VoltAmp[direction][side][component][planePosition],
							tfsf->m_VoltDelay[direction][side][component][planePosition],
							tfsf->m_VoltDelayDelta[direction][side][component][planePosition]);
					}

					unsigned int currentPosition[3] = {
						position[0], position[1], position[2]};
					if (side == 0)
						--currentPosition[direction];
					for (unsigned int component = 0; component < 2; ++component)
					{
						const unsigned int fieldComponent = component == 0 ? nP : nPP;
						append(current, fieldComponent, currentPosition,
							tfsf->m_CurrAmp[direction][side][component][planePosition],
							tfsf->m_CurrDelay[direction][side][component][planePosition],
							tfsf->m_CurrDelayDelta[direction][side][component][planePosition]);
					}
					++planePosition;
				}
			}
		}
	}

	Excitation* excitation = tfsf->m_Exc;
	if (excitation == NULL)
		return;
	const unsigned int signalLength = excitation->GetLength();
	unsigned int periodSteps = 0;
	if (excitation->GetSignalPeriod() > 0)
		periodSteps = static_cast<unsigned int>(
			excitation->GetSignalPeriod() / excitation->GetTimestep());

	auto upload = [signalLength, periodSteps](Impl::TFSFCorrectionSet& target,
		const HostCorrections& source, const FDTD_FLOAT* signal)
	{
		target.count = static_cast<unsigned int>(source.indices.size());
		target.signalLength = signalLength;
		target.periodSteps = periodSteps;
		DeviceAllocateAndCopy(target.indices, source.indices.data(), target.count);
		DeviceAllocateAndCopy(target.amplitudes, source.amplitudes.data(), target.count);
		DeviceAllocateAndCopy(target.delays, source.delays.data(), target.count);
		DeviceAllocateAndCopy(target.delayDeltas, source.delayDeltas.data(), target.count);
		DeviceAllocateAndCopy(target.signal, signal, signalLength);
	};

	// TFSF voltage corrections inject the incident H-field signal; current
	// corrections inject the incident E-field signal, matching Engine_Ext_TFSF.
	upload(m_Impl->tfsfVoltage, voltage, excitation->GetCurrentSignal());
	upload(m_Impl->tfsfCurrent, current, excitation->GetVoltageSignal());
}

void Engine_CUDA::UploadUPML(Operator_Ext_UPML* upml)
{
	m_Impl->pmlRegions.push_back(Impl::PMLRegion());
	Impl::PMLRegion& region = m_Impl->pmlRegions.back();
	for (unsigned int n = 0; n < 3; ++n)
	{
		region.start[n] = upml->m_StartPos[n];
		region.extent[n] = upml->m_numLines[n];
	}
	region.cells = static_cast<size_t>(region.extent[0])
		* region.extent[1] * region.extent[2];
	const size_t elements = 3 * region.cells;

	DeviceAllocateAndCopy(region.vv, upml->vv.data(), elements);
	DeviceAllocateAndCopy(region.vvfo, upml->vvfo.data(), elements);
	DeviceAllocateAndCopy(region.vvfn, upml->vvfn.data(), elements);
	DeviceAllocateAndCopy(region.ii, upml->ii.data(), elements);
	DeviceAllocateAndCopy(region.iifo, upml->iifo.data(), elements);
	DeviceAllocateAndCopy(region.iifn, upml->iifn.data(), elements);
	DeviceAllocate(region.voltageFlux, elements);
	DeviceAllocate(region.currentFlux, elements);
	CudaCheck(cudaMemset(region.voltageFlux, 0,
		elements * sizeof(FDTD_FLOAT)), "UPML voltage-flux initialization");
	CudaCheck(cudaMemset(region.currentFlux, 0,
		elements * sizeof(FDTD_FLOAT)), "UPML current-flux initialization");
}

void Engine_CUDA::UploadLumpedRLC(Operator_Ext_LumpedRLC* lumpedRLC)
{
	Impl::LumpedRLCData& data = m_Impl->lumpedRLC;
	data.count = lumpedRLC->RLC_count;
	if (data.count == 0)
		return;

	std::vector<size_t> indices(data.count);
	std::set<size_t> uniqueIndices;
	const size_t cells = m_Impl->cells;
	const unsigned int ny = numLines[1];
	const unsigned int nz = numLines[2];
	for (unsigned int n = 0; n < data.count; ++n)
	{
		indices[n] = static_cast<size_t>(lumpedRLC->v_RLC_dir[n]) * cells
			+ (static_cast<size_t>(lumpedRLC->v_RLC_pos[0][n]) * ny
				+ lumpedRLC->v_RLC_pos[1][n]) * nz
			+ lumpedRLC->v_RLC_pos[2][n];
		if (!uniqueIndices.insert(indices[n]).second)
			throw std::runtime_error(
				"CUDA engine does not support overlapping lumped-RLC cells");
	}

	DeviceAllocateAndCopy(data.indices, indices.data(), data.count);
	DeviceAllocateAndCopy(data.ilv, lumpedRLC->v_RLC_ilv, data.count);
	DeviceAllocateAndCopy(data.i2v, lumpedRLC->v_RLC_i2v, data.count);
	DeviceAllocateAndCopy(data.vv2, lumpedRLC->v_RLC_vv2, data.count);
	DeviceAllocateAndCopy(data.vj1, lumpedRLC->v_RLC_vj1, data.count);
	DeviceAllocateAndCopy(data.vj2, lumpedRLC->v_RLC_vj2, data.count);
	DeviceAllocateAndCopy(data.vvd, lumpedRLC->v_RLC_vvd, data.count);
	DeviceAllocateAndCopy(data.ib0, lumpedRLC->v_RLC_ib0, data.count);
	DeviceAllocateAndCopy(data.b1, lumpedRLC->v_RLC_b1, data.count);
	DeviceAllocateAndCopy(data.b2, lumpedRLC->v_RLC_b2, data.count);
	DeviceAllocate(data.inductorCurrent, data.count);
	CudaCheck(cudaMemset(data.inductorCurrent, 0,
		data.count * sizeof(double)), "lumped-RLC current initialization");
	for (unsigned int n = 0; n < 3; ++n)
	{
		DeviceAllocate(data.voltage[n], data.count);
		DeviceAllocate(data.current[n], data.count);
		CudaCheck(cudaMemset(data.voltage[n], 0,
			data.count * sizeof(double)),
			"lumped-RLC voltage-state initialization");
		CudaCheck(cudaMemset(data.current[n], 0,
			data.count * sizeof(double)),
			"lumped-RLC current-state initialization");
	}
}

void Engine_CUDA::UploadExcitation(Operator_Ext_Excitation* excitation)
{
	const size_t cells = m_Impl->cells;
	const unsigned int ny = numLines[1];
	const unsigned int nz = numLines[2];
	Excitation* signal = excitation->m_Exc;
	if (signal == NULL)
		return;

	const unsigned int signalLength = signal->GetLength();
	unsigned int periodSteps = 0;
	if (signal->GetSignalPeriod() > 0)
		periodSteps = static_cast<unsigned int>(
			signal->GetSignalPeriod() / signal->GetTimestep());

	Impl::ExcitationSet& voltage = m_Impl->voltageExcitation;
	voltage.count = excitation->Volt_Count;
	voltage.signalLength = signalLength;
	voltage.periodSteps = periodSteps;
	std::vector<size_t> voltageIndices(voltage.count);
	for (unsigned int n = 0; n < voltage.count; ++n)
	{
		voltageIndices[n] = static_cast<size_t>(excitation->Volt_dir[n]) * cells
			+ (static_cast<size_t>(excitation->Volt_index[0][n]) * ny
				+ excitation->Volt_index[1][n]) * nz
			+ excitation->Volt_index[2][n];
	}
	DeviceAllocateAndCopy(voltage.indices, voltageIndices.data(), voltage.count);
	DeviceAllocateAndCopy(voltage.amplitudes, excitation->Volt_amp, voltage.count);
	DeviceAllocateAndCopy(voltage.delays, excitation->Volt_delay, voltage.count);
	DeviceAllocateAndCopy(voltage.signal, signal->GetVoltageSignal(), signalLength);

	Impl::ExcitationSet& current = m_Impl->currentExcitation;
	current.count = excitation->Curr_Count;
	current.signalLength = signalLength;
	current.periodSteps = periodSteps;
	std::vector<size_t> currentIndices(current.count);
	for (unsigned int n = 0; n < current.count; ++n)
	{
		currentIndices[n] = static_cast<size_t>(excitation->Curr_dir[n]) * cells
			+ (static_cast<size_t>(excitation->Curr_index[0][n]) * ny
				+ excitation->Curr_index[1][n]) * nz
			+ excitation->Curr_index[2][n];
	}
	DeviceAllocateAndCopy(current.indices, currentIndices.data(), current.count);
	DeviceAllocateAndCopy(current.amplitudes, excitation->Curr_amp, current.count);
	DeviceAllocateAndCopy(current.delays, excitation->Curr_delay, current.count);
	DeviceAllocateAndCopy(current.signal, signal->GetCurrentSignal(), signalLength);
}

bool Engine_CUDA::IterateTS(unsigned int iterTS)
{
	if (!m_Initialized)
		throw std::runtime_error("CUDA engine is not initialized");

	const unsigned int threads = 256;
	const size_t interiorCells =
		(numLines[0] > 1 && numLines[1] > 1 && numLines[2] > 1)
		? static_cast<size_t>(numLines[0] - 1)
			* (numLines[1] - 1) * (numLines[2] - 1)
		: 0;

	for (unsigned int iter = 0; iter < iterTS; ++iter)
	{
		Impl::LumpedRLCData& lumpedRLC = m_Impl->lumpedRLC;
		if (lumpedRLC.count != 0)
		{
			double* temporary = lumpedRLC.voltage[2];
			lumpedRLC.voltage[2] = lumpedRLC.voltage[1];
			lumpedRLC.voltage[1] = lumpedRLC.voltage[0];
			lumpedRLC.voltage[0] = temporary;
			LumpedRLCPreVoltageKernel<<<
				BlockCount(lumpedRLC.count, threads), threads>>>(
				lumpedRLC.inductorCurrent, lumpedRLC.voltage[1],
				lumpedRLC.i2v, lumpedRLC.ilv, lumpedRLC.count);
			CudaCheck(cudaGetLastError(),
				"lumped-RLC pre-voltage kernel launch");
		}

		for (size_t n = 0; n < m_Impl->pmlRegions.size(); ++n)
		{
			Impl::PMLRegion& region = m_Impl->pmlRegions[n];
			UPMLPreKernel<<<BlockCount(region.cells, threads), threads>>>(
				m_Impl->volt, region.voltageFlux, region.vv, region.vvfo,
				region.start[0], region.start[1], region.start[2],
				region.extent[0], region.extent[1], region.extent[2],
				numLines[0], numLines[1], numLines[2], region.cells);
			CudaCheck(cudaGetLastError(), "UPML pre-voltage kernel launch");
		}

		UpdateVoltagesKernel<<<BlockCount(m_Impl->cells, threads), threads>>>(
			m_Impl->volt, m_Impl->curr, m_Impl->vv, m_Impl->vi,
			numLines[0], numLines[1], numLines[2], m_Impl->cells);
		CudaCheck(cudaGetLastError(), "voltage kernel launch");

		for (size_t n = 0; n < m_Impl->pmlRegions.size(); ++n)
		{
			Impl::PMLRegion& region = m_Impl->pmlRegions[n];
			UPMLPostKernel<<<BlockCount(region.cells, threads), threads>>>(
				m_Impl->volt, region.voltageFlux, region.vvfn,
				region.start[0], region.start[1], region.start[2],
				region.extent[0], region.extent[1], region.extent[2],
				numLines[0], numLines[1], numLines[2], region.cells);
			CudaCheck(cudaGetLastError(), "UPML post-voltage kernel launch");
		}

		Impl::TFSFCorrectionSet& tfsfVoltage = m_Impl->tfsfVoltage;
		if (tfsfVoltage.count != 0)
		{
			ApplyTFSFCorrectionKernel<<<
				BlockCount(tfsfVoltage.count, threads), threads>>>(
					m_Impl->volt, tfsfVoltage.indices, tfsfVoltage.amplitudes,
					tfsfVoltage.delays, tfsfVoltage.delayDeltas,
					tfsfVoltage.count, tfsfVoltage.signal,
					tfsfVoltage.signalLength, tfsfVoltage.periodSteps, numTS);
			CudaCheck(cudaGetLastError(), "TFSF voltage kernel launch");
		}

		UpdateSteadyState(threads);

		if (lumpedRLC.count != 0)
		{
			double* temporary = lumpedRLC.current[2];
			lumpedRLC.current[2] = lumpedRLC.current[1];
			lumpedRLC.current[1] = lumpedRLC.current[0];
			lumpedRLC.current[0] = temporary;
			LumpedRLCApplyVoltageKernel<<<
				BlockCount(lumpedRLC.count, threads), threads>>>(
				m_Impl->volt, lumpedRLC.indices, lumpedRLC.inductorCurrent,
				lumpedRLC.voltage[0], lumpedRLC.voltage[1],
				lumpedRLC.voltage[2], lumpedRLC.current[0],
				lumpedRLC.current[1], lumpedRLC.current[2],
				lumpedRLC.vv2, lumpedRLC.vj1, lumpedRLC.vj2,
				lumpedRLC.vvd, lumpedRLC.ib0, lumpedRLC.b1,
				lumpedRLC.b2, lumpedRLC.count);
			CudaCheck(cudaGetLastError(),
				"lumped-RLC voltage kernel launch");
		}

		Impl::ExcitationSet& voltage = m_Impl->voltageExcitation;
		if (voltage.count != 0)
		{
			ApplyExcitationKernel<<<BlockCount(voltage.count, threads), threads>>>(
				m_Impl->volt, voltage.indices, voltage.amplitudes,
				voltage.delays, voltage.count, voltage.signal,
				voltage.signalLength, voltage.periodSteps, numTS);
			CudaCheck(cudaGetLastError(), "voltage excitation kernel launch");
		}

		for (size_t n = 0; n < m_Impl->pmlRegions.size(); ++n)
		{
			Impl::PMLRegion& region = m_Impl->pmlRegions[n];
			UPMLPreKernel<<<BlockCount(region.cells, threads), threads>>>(
				m_Impl->curr, region.currentFlux, region.ii, region.iifo,
				region.start[0], region.start[1], region.start[2],
				region.extent[0], region.extent[1], region.extent[2],
				numLines[0], numLines[1], numLines[2], region.cells);
			CudaCheck(cudaGetLastError(), "UPML pre-current kernel launch");
		}

		if (interiorCells != 0)
		{
			UpdateCurrentsKernel<<<BlockCount(interiorCells, threads), threads>>>(
				m_Impl->curr, m_Impl->volt, m_Impl->ii, m_Impl->iv,
				numLines[0], numLines[1], numLines[2], interiorCells);
			CudaCheck(cudaGetLastError(), "current kernel launch");
		}

		for (size_t n = 0; n < m_Impl->pmlRegions.size(); ++n)
		{
			Impl::PMLRegion& region = m_Impl->pmlRegions[n];
			UPMLPostKernel<<<BlockCount(region.cells, threads), threads>>>(
				m_Impl->curr, region.currentFlux, region.iifn,
				region.start[0], region.start[1], region.start[2],
				region.extent[0], region.extent[1], region.extent[2],
				numLines[0], numLines[1], numLines[2], region.cells);
			CudaCheck(cudaGetLastError(), "UPML post-current kernel launch");
		}

		Impl::TFSFCorrectionSet& tfsfCurrent = m_Impl->tfsfCurrent;
		if (tfsfCurrent.count != 0)
		{
			ApplyTFSFCorrectionKernel<<<
				BlockCount(tfsfCurrent.count, threads), threads>>>(
					m_Impl->curr, tfsfCurrent.indices, tfsfCurrent.amplitudes,
					tfsfCurrent.delays, tfsfCurrent.delayDeltas,
					tfsfCurrent.count, tfsfCurrent.signal,
					tfsfCurrent.signalLength, tfsfCurrent.periodSteps, numTS);
			CudaCheck(cudaGetLastError(), "TFSF current kernel launch");
		}

		Impl::ExcitationSet& current = m_Impl->currentExcitation;
		if (current.count != 0)
		{
			ApplyExcitationKernel<<<BlockCount(current.count, threads), threads>>>(
				m_Impl->curr, current.indices, current.amplitudes,
				current.delays, current.count, current.signal,
				current.signalLength, current.periodSteps, numTS);
			CudaCheck(cudaGetLastError(), "current excitation kernel launch");
		}

		if (m_DebugSync)
			CudaCheck(cudaDeviceSynchronize(), "debug synchronization");
		++numTS;
	}
	return true;
}

void Engine_CUDA::PrepareProcessing()
{
	if (!m_Initialized)
		return;
	CudaCheck(cudaDeviceSynchronize(), "processing synchronization");
	const size_t bytes = m_Impl->fieldElements * sizeof(FDTD_FLOAT);
	CudaCheck(cudaMemcpy(volt_ptr->data(), m_Impl->volt, bytes,
		cudaMemcpyDeviceToHost), "voltage processing copy");
	CudaCheck(cudaMemcpy(curr_ptr->data(), m_Impl->curr, bytes,
		cudaMemcpyDeviceToHost), "current processing copy");
}
