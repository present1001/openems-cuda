/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "processfields_fd.h"
#include "Common/operator_base.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include <climits>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

ProcessFieldsFD::ProcessFieldsFD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	m_FirstSample = 1;
	m_LastSample = UINT_MAX;
	m_PeriodicAveraging = false;
	m_Normalized = false;
}

ProcessFieldsFD::~ProcessFieldsFD()
{
	for (size_t n = 0; n<m_FD_Fields.size(); ++n)
	{
		delete m_FD_Fields.at(n);
	}
	m_FD_Fields.clear();
}

void ProcessFieldsFD::InitProcess()
{
	if (Enabled==false) return;

	if (m_FD_Samples.size()==0)
	{
		cerr << "ProcessFieldsFD::InitProcess: No frequencies found... skipping this dump!" << endl;
		Enabled=false;
		return;
	}

	//setup the hdf5 file
	ProcessFields::InitProcess();

	if (m_Vtk_Dump_File)
		m_Vtk_Dump_File->SetHeader(string("openEMS FD Field Dump -- Interpolation: ")+m_Eng_Interface->GetInterpolationTypeString());

	if (m_HDF5_Dump_File)
	{
		m_HDF5_Dump_File->SetCurrentGroup("/FieldData/FD");
		m_HDF5_Dump_File->WriteAttribute("/FieldData/FD","frequency",m_FD_Samples);
	}

	//create data structures...
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		ArrayLib::ArrayNIJK<std::complex<double>>* field_fd = new ArrayLib::ArrayNIJK<std::complex<double>>("FD field", numLines);
		m_FD_Fields.push_back(field_fd);
	}
}

void ProcessFieldsFD::SetSampleWindow(unsigned int first, unsigned int last)
{
	m_FirstSample = first ? first : 1u;
	m_LastSample = last<m_FirstSample ? m_FirstSample : last;
	const unsigned long long firstTS =
		static_cast<unsigned long long>(m_FirstSample-1) * m_FD_Interval;
	startTS = firstTS>UINT_MAX ? UINT_MAX : static_cast<unsigned int>(firstTS);
	if (m_LastSample==UINT_MAX)
		stopTS = UINT_MAX;
	else
	{
		const unsigned long long lastTS =
			static_cast<unsigned long long>(m_LastSample-1) * m_FD_Interval;
		stopTS = lastTS>UINT_MAX ? UINT_MAX : static_cast<unsigned int>(lastTS);
	}
}

int ProcessFieldsFD::Process()
{
	if (Enabled==false) return -1;
	if (CheckTimestep()==false) return GetNextInterval();

	if ((m_FD_Interval==0) || (m_Eng_Interface->GetNumberOfTimesteps()%m_FD_Interval!=0))
		return GetNextInterval();

	const unsigned int sample =
		m_Eng_Interface->GetNumberOfTimesteps()/m_FD_Interval + 1;
	if ((sample<m_FirstSample) || (sample>m_LastSample))
		return GetNextInterval();

	ArrayLib::ArrayNIJK<FDTD_FLOAT> tmp_field_td;
	if (!CalcField(tmp_field_td))
		return -1;
	FDTD_FLOAT* field_td = tmp_field_td.data();

	std::complex<double>* field_fd = NULL;

	double T = m_Eng_Interface->GetTime(m_dualTime);
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		std::complex<double> exp_jwt_2_dt = std::exp(
			-2.0 * I_UNIT * PI * m_FD_Samples.at(n) * T);
		if (!m_PeriodicAveraging)
		{
			exp_jwt_2_dt *= 2; // *2 for single-sided spectrum
			exp_jwt_2_dt *= Op->GetTimestep() * m_FD_Interval; // multiply with timestep-interval
		}
		unsigned int N = m_FD_Fields.at(n)->size();
		field_fd = m_FD_Fields.at(n)->data();
		for (unsigned int ijk=0;ijk<N;++ijk)
			field_fd[ijk] += static_cast<double>(field_td[ijk]) * exp_jwt_2_dt;
	}
	++m_FD_SampleCount;
	return GetNextInterval();
}

void ProcessFieldsFD::PostProcess()
{
	if (m_PeriodicAveraging && !m_Normalized && (m_FD_SampleCount!=0))
	{
		const double scale = 1.0 / m_FD_SampleCount;
		for (size_t n=0; n<m_FD_Fields.size(); ++n)
		{
			std::complex<double>* field = m_FD_Fields.at(n)->data();
			const unsigned int count = m_FD_Fields.at(n)->size();
			for (unsigned int index=0; index<count; ++index)
				field[index] *= scale;
		}
		m_Normalized = true;
	}
	DumpFDData();
}

void ProcessFieldsFD::DumpFDData()
{
	if (m_fileType==VTK_FILETYPE)
	{
		ArrayLib::ArrayNIJK<float> tmp_field("field", numLines);
		std::complex<double>* field_fd = NULL;
		float* field = tmp_field.data();
		double angle=0;
		int Nr_Ph = 21;

		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			std::string str_freq;
			double freq = m_FD_Samples.at(n);
			if ((freq-long(freq))==0)
				// if the frequency is a natural number, export as long
				str_freq = std::to_string(long(freq));
			else
				str_freq = std::to_string(freq);

			unsigned int N = m_FD_Fields.at(n)->size();
			field_fd = m_FD_Fields.at(n)->data();
			//dump multiple phase to vtk-files
			for (int p=0; p<Nr_Ph; ++p)
			{
				angle = 2.0 * PI * p / Nr_Ph;
				std::complex<double> exp_jwt = std::exp(I_UNIT * angle);
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = real(field_fd[ijk] * exp_jwt);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_p=" << std::setw( 3 ) << std::setfill( '0' ) <<(int)(angle * 180 / PI);

				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}

			{
				//dump magnitude to vtk-files
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = abs(field_fd[ijk]);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_abs";
				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}

			{
				//dump phase to vtk-files
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = arg(field_fd[ijk]);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_arg";
				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}
		}
		return;
	}

	if (m_fileType==HDF5_FILETYPE)
	{
		m_HDF5_Dump_File->WriteAttribute(
			"/FieldData/FD", "sample_count", m_FD_SampleCount);
		m_HDF5_Dump_File->WriteAttribute(
			"/FieldData/FD", "sample_interval", m_FD_Interval);
		m_HDF5_Dump_File->WriteAttribute(
			"/FieldData/FD", "first_sample", m_FirstSample);
		m_HDF5_Dump_File->WriteAttribute(
			"/FieldData/FD", "last_sample", m_LastSample);
		m_HDF5_Dump_File->WriteAttribute(
			"/FieldData/FD", "normalization",
			std::string(m_PeriodicAveraging
				? "periodic-average" : "single-sided-integral"));
		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			stringstream ss;
			ss << "f" << n;
			// size_t datasize[]={numLines[0],numLines[1],numLines[2]};
			if (m_HDF5_Dump_File->WriteVectorField<std::complex<double>>(ss.str(), *m_FD_Fields.at(n), g_settings.GetLegacyHDF5Dumps())==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;

			if (g_settings.GetLegacyHDF5Dumps())
			{
				if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str()+"_real", "frequency", m_FD_Samples.at(n))==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
				if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str()+"_imag", "frequency", m_FD_Samples.at(n))==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
			}
			else if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str(), "frequency", m_FD_Samples.at(n))==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
		}
		return;
	}

	cerr << "ProcessFieldsFD::Process: unknown File-Type" << endl;
}
