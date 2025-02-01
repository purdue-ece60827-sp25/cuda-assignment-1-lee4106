
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x; //calculate thread id
	if (i < size) {
		y[i] = scale * x[i] + y[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	float scale = (float)(rand() % 100);
	int size = vectorSize * sizeof(float);
	float * a, * b, * c;
	a = (float *) malloc(size);
	b = (float *) malloc(size);
	c = (float *) malloc(size);
	if (a == NULL || b == NULL || c== NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}
	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);


	float * a_d, * c_d;
	cudaMalloc((void **) &a_d, size);
	cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &c_d, size);
	cudaMemcpy(c_d, b, size, cudaMemcpyHostToDevice);

	//calling GPU device Kernal
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(a_d, c_d, scale, vectorSize);

	cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost); //move data back to host mem

	//Verify
	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	cudaFree(a_d);
	cudaFree(c_d);
	free(c);
	free(a);
	free(b);
	std::cout << "Part A Done!\n";
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	
	int i = threadIdx.x + blockDim.x * blockIdx.x; //calculate thread id
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);
	if (i < pSumSize){
		pSums[i] = 0;
		for (uint64_t j = 0; j < sampleSize; j++){
			float x = curand_uniform(&rng);
			float y = curand_uniform(&rng);
			if ((x * x + y * y) <= 1.0f ){
				pSums[i] += 1;
			}
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int i = threadIdx.x + blockDim.x * blockIdx.x; //calculate thread id
	if (i < pSumSize/reduceSize) {
		totals[i] = 0;
		for (uint64_t j = 0; j < reduceSize; j ++){
			if (j+i*reduceSize < pSumSize) {
				totals[i] += pSums[j+i*reduceSize];
			}
		}
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here

	//cal dim
	uint64_t TB_size = 256;
	dim3 DimGrid(ceil(generateThreadCount/TB_size),1,1);
	dim3 DimBlock(TB_size,1,1);


	uint64_t * pSums, * totals;
	uint64_t psize = generateThreadCount*sizeof(uint64_t);
	uint64_t tsize = reduceThreadCount*sizeof(uint64_t);

	pSums = (uint64_t *) malloc(psize);
	totals = (uint64_t *) malloc(tsize);
	if (pSums==NULL || totals==NULL){
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}


	uint64_t * pSums_d, * totals_d;

	////malloc and calling device generatepoint kernal
	cudaMalloc((void **) &pSums_d, psize);
	//cudaMemcpy(pSums_d, pSums, psize, cudaMemcpyHostToDevice);
	generatePoints<<<DimGrid, DimBlock>>>(pSums_d, generateThreadCount, sampleSize);

	//malloc and calling device reduction kernal
	cudaMalloc((void **) &totals_d, tsize);
	//cudaMemcpy(totals_d, totals, tsize, cudaMemcpyHostToDevice);
	reduceCounts<<<DimGrid, DimBlock>>>(pSums_d,totals_d,generateThreadCount, reduceSize);

	cudaMemcpy(totals, totals_d, tsize, cudaMemcpyDeviceToHost); //move data back to host mem
	

	uint64_t hit_sum = 0;
	for (uint64_t i = 0; i<reduceThreadCount; i++){
		hit_sum += totals[i];
	}
	approxPi = (double)hit_sum/(generateThreadCount*sampleSize) *4;

	//free all the malloc
	cudaFree(pSums_d);
	cudaFree(totals_d);
	free(pSums);
	free(totals);

	return approxPi;
}
