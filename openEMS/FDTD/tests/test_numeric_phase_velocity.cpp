#include "FDTD/numeric_phase_velocity.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

double Expected(double spacing, double c0, double dT, double freq)
{
	const double pi = std::acos(-1.0);
	const double k = std::asin(
		spacing / c0 / dT * std::sin(pi * freq * dT)) * 2.0 / spacing;
	return 2.0 * pi * freq / k;
}

bool NearlyEqual(double lhs, double rhs)
{
	return std::fabs(lhs - rhs) <= 1e-12 * std::fabs(rhs);
}

} // namespace

int main()
{
	const double spacing[3] = {0.1, 0.2, 0.3};
	const double directions[3][3] = {
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, -1.0}
	};
	const double c0 = 1.0;
	const double dT = 0.1;
	const double freq = 0.5;

	double velocities[3] = {0.0, 0.0, 0.0};
	for (unsigned int n = 0; n < 3; ++n)
	{
		if (!openems::CalcCartesianNumericPhaseVelocity(
				spacing, directions[n], c0, dT, freq, velocities[n]))
		{
			std::cerr << "Cartesian direction " << n << " was not detected\n";
			return EXIT_FAILURE;
		}
		if (!NearlyEqual(velocities[n], Expected(spacing[n], c0, dT, freq)))
		{
			std::cerr << "Direction " << n << " used the wrong mesh spacing\n";
			return EXIT_FAILURE;
		}
	}

	if ((velocities[0] == velocities[1]) || (velocities[1] == velocities[2]))
	{
		std::cerr << "Anisotropic spacings unexpectedly produced equal velocities\n";
		return EXIT_FAILURE;
	}

	const double diagonal[3] = {0.5, 0.5, 0.0};
	double unused = 0.0;
	if (openems::CalcCartesianNumericPhaseVelocity(
			spacing, diagonal, c0, dT, freq, unused))
	{
		std::cerr << "A non-Cartesian direction was accepted\n";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
