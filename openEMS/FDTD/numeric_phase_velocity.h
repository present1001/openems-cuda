/*
 * Copyright (C) 2026 openEMS contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef NUMERIC_PHASE_VELOCITY_H
#define NUMERIC_PHASE_VELOCITY_H

#include <cmath>

namespace openems
{

// Return true and set phaseVelocity when propDir describes a Cartesian axis.
// Keeping this calculation independent from Operator makes the anisotropic
// mesh behavior directly testable.
inline bool CalcCartesianNumericPhaseVelocity(
	const double meshDisc[3],
	const double propDir[3],
	double c0,
	double dT,
	double freq,
	double& phaseVelocity)
{
	const double pi = std::acos(-1.0);
	for (unsigned int n = 0; n < 3; ++n)
	{
		const unsigned int nP = (n + 1) % 3;
		const unsigned int nPP = (n + 2) % 3;
		if ((std::fabs(propDir[n]) == 1.0) &&
			(propDir[nP] == 0.0) && (propDir[nPP] == 0.0))
		{
			const double spacing = meshDisc[n];
			const double waveNumber =
				std::asin(spacing / c0 / dT * std::sin(pi * freq * dT))
				* 2.0 / spacing;
			phaseVelocity = 2.0 * pi * freq / waveNumber;
			return true;
		}
	}
	return false;
}

} // namespace openems

#endif // NUMERIC_PHASE_VELOCITY_H
