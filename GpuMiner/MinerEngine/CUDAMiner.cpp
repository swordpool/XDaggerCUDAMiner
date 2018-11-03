#include "CUDAMiner.h"
#include "Hash/sha256_mod.h"

using namespace XDag;

#define MAX_GPU_ERROR_COUNT 3
#define OUTPUT_SIZE 15  //15 positions in output buffer + 1 position for flag


uint32_t CUDAMiner::_numInstances = 0;
int CUDAMiner::_devices[MAX_CUDA_DEVICES] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

CUDAMiner::CUDAMiner(uint32_t index, XTaskProcessor* taskProcessor)
	:Miner("cuda-", index, taskProcessor)
{
}


CUDAMiner::~CUDAMiner()
{
}

bool CUDAMiner::ConfigureCuda(
	uint32_t localWorkSize,
	uint32_t globalWorkSizeMultiplier,
	uint32_t platformId,
	bool useOpenClCpu
)
{
	//TODO: do I need automatically detemine path to executable folder?
	int devicesCnt = 0;
	cudaError err = cudaGetDeviceCount(&devicesCnt);
	if (err != cudaSuccess)
	{
		printf("No CUDA devices found.\n");
		return false;
	}
	printf("Found CUDA devices:\n");
	for (int i = 0; i<devicesCnt; ++i)
	{
		cudaDeviceProp prop;
		err = cudaGetDeviceProperties(&prop, i);
		if (err == cudaSuccess) {
			std::string name = prop.name;
			printf(" %s with %zu bytes of memory.\n", prop.name,prop.totalGlobalMem);
		}
	}

	return true;
}

uint32_t CUDAMiner::GetNumDevices()
{
	int cudaDevices = 0;
	cudaError err = cudaGetDeviceCount(&cudaDevices);
	return err == cudaSuccess ? cudaDevices : 0;
}

bool CUDAMiner::Initialize()
{
	// get all platforms
	// get all platforms
	try
	{
		// get GPU device of the CUDA
		int devices = 0;
		cudaError err = cudaGetDeviceCount(&devices);
		if (err != cudaSuccess || devices <=0 )
		{
			printf("No CUDA devices found.\n");
			return false;
		}

		//AddDefinition(_kernelCode, "PLATFORM", platformId);
		//AddDefinition(_kernelCode, "OUTPUT_SIZE", OUTPUT_SIZE);
		//AddDefinition(_kernelCode, "ITERATIONS_COUNT", KERNEL_ITERATIONS);
		err = cudaSetDevice(0);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			return false;
		}

		// create _stateBuffer buffers
		printf("Creating _stateBuffer buffer\n");
		err = cudaMalloc(&_stateBuffer, 32);
		if (err != cudaSuccess) {
			cwarn << "Alloc _stateBuffer failed.";
			return false;
		}

		// create _precalcStateBuffer buffers
		printf("Creating precalcState buffer\n");
		err = cudaMalloc(&_precalcStateBuffer, 32);
		if (err != cudaSuccess) {
			cwarn << "Alloc _precalcStateBuffer failed.";
			return false;
		}

		// create _precalcStateBuffer buffers
		printf("Creating dataBuffer buffer\n");
		err = cudaMalloc(&_dataBuffer, 56);
		if (err != cudaSuccess) {
			cwarn << "Alloc _dataBuffer failed.";
			return false;
		}

		// create mining buffers
		printf("Creating output buffer\n");
		err = cudaMalloc(&_searchBuffer, (OUTPUT_SIZE + 1) * sizeof(uint64_t));
		if (err != cudaSuccess) {
			cwarn << "Alloc _searchBuffer failed.";
			return false;
		}

		printf("Creating targetH buffer\n");
		err = cudaMalloc(&_targetH, 4);
		if (err != cudaSuccess) {
			cwarn << "Alloc _targetH failed.";
			return false;
		}

		printf("Creating targetG buffer\n");
		err = cudaMalloc(&_targetG, 4);
		if (err != cudaSuccess) {
			cwarn << "Alloc _targetG failed.";
			return false;
		}
	}
	catch (...)
	{
		printf("CUDA init failed\n");
		return false;
	}
	return true;
}

HwMonitor CUDAMiner::Hwmon()
{
	HwMonitor hw;
	unsigned int tempC = 0, fanpcnt = 0;
	hw.tempC = tempC;
	hw.fanP = fanpcnt;
	return hw;
}

void CUDAMiner::WorkLoop()
{
	int errorCount = 0;
	while (errorCount < MAX_GPU_ERROR_COUNT)
	{
		try
		{
			InternalWorkLook(errorCount);
			break;
		}
		catch (...)
		{
			if (++errorCount < MAX_GPU_ERROR_COUNT)
			{
				if (!Reset())
				{
					break;
				}
			}
		}
	}
}
void call_kernel(
	uint64_t startNonce,
	uint32_t* state,
	uint32_t* precalcState,
	uint32_t* data,
	uint32_t* targetH,
	uint32_t* targetG,
	uint64_t* output,
	uint32_t throughput);


void CUDAMiner::InternalWorkLook(int& errorCount)
{
	xdag_field last;
	uint64_t prevTaskIndex = 0;
	uint64_t nonce = 0;
	uint32_t loopCounter = 0;

	uint64_t results[OUTPUT_SIZE + 1];
	uint64_t zeroBuffer[OUTPUT_SIZE + 1];
	memset(zeroBuffer, 0, (OUTPUT_SIZE + 1) * sizeof(uint64_t));

	while (!ShouldStop())
	{
		XTaskWrapper* taskWrapper = GetTask();
		if (taskWrapper == NULL)
		{
			printf("No work. Pause for 3 s.\n");
			std::this_thread::sleep_for(std::chrono::seconds(3));
			continue;
		}

		if (taskWrapper->GetIndex() != prevTaskIndex)
		{
			//new task came, we have to finish current task and reload all data
			if (prevTaskIndex > 0)
			{
				cudaDeviceSynchronize();
			}

			prevTaskIndex = taskWrapper->GetIndex();
			loopCounter = 0;
			memcpy(last.data, taskWrapper->GetTask()->nonce.data, sizeof(xdag_hash_t));
			nonce = last.amount + _index * 1000000000000;//TODO: think of nonce increment

			WriteKernelArgs(taskWrapper, zeroBuffer);
		}

		bool hasSolution = false;
		if (loopCounter > 0)
		{
			// Read results.
			ReadData(results);
			errorCount = 0;

			//miner return an array with 16 64-bit values. If nonce for hash lower than target hash is found - it is written to array. 
			//the first value in array contains count of found solutions
			hasSolution = results[0] > 0;
			if (hasSolution)
			{
				// Reset search buffer if any solution found.
				cudaMemcpy(_searchBuffer, zeroBuffer, sizeof(zeroBuffer), cudaMemcpyHostToDevice);
			}
		}

		const uint32_t throughput = 16777216;
		// Run the kernel.
		//_searchKernel.setArg(KERNEL_ARG_NONCE, nonce);
		//_queue.enqueueNDRangeKernel(_searchKernel, cl::NullRange, _globalWorkSize, _workgroupSize);
		call_kernel(nonce,
			(uint32_t*)_stateBuffer,
			(uint32_t*)_precalcStateBuffer,
			(uint32_t*)_dataBuffer,
			(uint32_t*)_targetH,
			(uint32_t*)_targetG,
			(uint64_t*)_searchBuffer,
			throughput);

		// Report results while the kernel is running.
		// It takes some time because hashes must be re-evaluated on CPU.
		if (hasSolution)
		{
			//we need to recalculate hashes for all founded nonces and choose the minimal one
			SetMinShare(taskWrapper, results, last);
			//new minimal hash is written as target hash for GPU
			cudaError err = cudaMemcpy(_targetH, &((uint32_t*)taskWrapper->GetTask()->minhash.data)[7], 4, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				cwarn << " Write _targetH failed.";
			}
			err = cudaMemcpy(_targetG, &((uint32_t*)taskWrapper->GetTask()->minhash.data)[6], 4, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				cwarn << " Write _targetG failed.";
			}
		}

		uint32_t hashesProcessed = throughput * 16;
		//if (_useVectors)
		//{
		//	hashesProcessed <<= 1;
		//}

		// Increase start nonce for following kernel execution.
		nonce += hashesProcessed;
		// Report hash count
		AddHashCount(hashesProcessed);
		++loopCounter;
	}

	// Make sure the last buffer write has finished --
	// it reads local variable.
	cudaDeviceSynchronize();
}

void CUDAMiner::WriteKernelArgs(XTaskWrapper* taskWrapper, uint64_t* zeroBuffer)
{
	// Update constant buffers.
	cudaError cudaStatus = cudaMemcpy(_stateBuffer, taskWrapper->GetTask()->ctx.state, 32,cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cwarn << " Write Constants failed.";
	}

	cudaStatus = cudaMemcpy(_precalcStateBuffer, taskWrapper->GetPrecalcState(), 32, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cwarn << " Write Constants failed.";
	}

	cudaStatus = cudaMemcpy(_dataBuffer, taskWrapper->GetReversedData(), 56, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cwarn << " Write Constants failed.";
	}

	cudaStatus = cudaMemcpy(_searchBuffer, zeroBuffer, sizeof(zeroBuffer), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cwarn << " Write _searchBuffer ZERO failed.";
	}

	cudaStatus = cudaMemcpy(_targetH, &((uint32_t*)taskWrapper->GetTask()->minhash.data)[7], 4, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cwarn << " Write _targetH failed.";
	}
	cudaStatus = cudaMemcpy(_targetG, &((uint32_t*)taskWrapper->GetTask()->minhash.data)[6], 4, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cwarn << " Write _targetG failed.";
	}
}

bool CUDAMiner::Reset()
{
	printf("GPU will be restarted\n");

	// pause for 0.5 sec
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	try
	{
		if (_searchBuffer) cudaFree(_searchBuffer);
		
		return Initialize();
	}
	catch (...)
	{
	}
	return false;
}

void CUDAMiner::ReadData(uint64_t* results)
{
	cudaError cudaStatus = cudaDeviceSynchronize();
	if ( cudaStatus != cudaSuccess) {
		cwarn << "Get cudaDeviceSynchronize failed - " << cudaGetErrorString(cudaStatus);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	cudaStatus = cudaMemcpy(results, _searchBuffer, (OUTPUT_SIZE + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cwarn << "cudaMemcpy failed - " << cudaGetErrorString(cudaStatus);
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
}

void CUDAMiner::SetMinShare(XTaskWrapper* taskWrapper, uint64_t* searchBuffer, xdag_field& last)
{
	xdag_hash_t minHash;
	xdag_hash_t currentHash;
	uint64_t minNonce = 0;

	uint32_t size = searchBuffer[0] < OUTPUT_SIZE ? (uint32_t)searchBuffer[0] : OUTPUT_SIZE;
	for (uint32_t i = 1; i <= size; ++i)
	{
		uint64_t nonce = searchBuffer[i];
		if (nonce == 0)
		{
			continue;
		}
		shamod::shasha(taskWrapper->GetTask()->ctx.state, 
			taskWrapper->GetTask()->ctx.data, 
			nonce, 
			(uint8_t*)currentHash);

		if (!minNonce || XHash::CompareHashes(currentHash, minHash) < 0)
		{
			memcpy(minHash, currentHash, sizeof(xdag_hash_t));
			minNonce = nonce;
		}
	}

#ifdef _DEBUG
	assert(minNonce > 0);
#endif
	if (minNonce > 0)
	{
		last.amount = minNonce;
		taskWrapper->SetShare(last.data, minHash);
	}
}