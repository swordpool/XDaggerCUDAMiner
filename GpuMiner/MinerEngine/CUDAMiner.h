#pragma once
#include <cuda.h>  
#include "Core/Worker.h"
#include "Core/Miner.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define MAX_CUDA_DEVICES 16

namespace XDag
{
	class CUDAMiner : public Miner
	{
	public:
		CUDAMiner(uint32_t index, XTaskProcessor* taskProcessor);
		~CUDAMiner();
	public:
		static uint32_t Instances() { return _numInstances > 0 ? _numInstances : 1; }
		static uint32_t GetNumDevices();
		static bool ConfigureCuda(
			uint32_t localWorkSize,
			uint32_t globalWorkSizeMultiplier,
			uint32_t platformId,
			bool useOpenClCpu
		);
		static void SetNumInstances(uint32_t instances) { _numInstances = std::min<uint32_t>(instances, GetNumDevices()); }
		static void SetDevices(uint32_t * devices, uint32_t selectedDeviceCount)
		{
			for (uint32_t i = 0; i < selectedDeviceCount; i++)
			{
				_devices[i] = devices[i];
			}
		}

		bool Initialize() override;
		HwMonitor Hwmon() override;
		void InternalWorkLook(int& errorCount);
	private:
		void WorkLoop() override;
		bool Reset();
		void ReadData(uint64_t* results);
		void SetMinShare(XTaskWrapper* taskWrapper, uint64_t* searchBuffer, xdag_field& last);
		void WriteKernelArgs(XTaskWrapper* taskWrapper, uint64_t* zeroBuffer);
		static int _devices[MAX_CUDA_DEVICES];
		static uint32_t _numInstances;

		void * _stateBuffer;
		void * _precalcStateBuffer;
		void * _dataBuffer;
		void * _searchBuffer;
		void * _targetH;
		void * _targetG;
	};
}
