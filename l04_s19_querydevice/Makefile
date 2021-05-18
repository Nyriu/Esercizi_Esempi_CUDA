all:
	@#nvcc -std=c++11 main.cu -o main
	@#nvcc -o main main.cu
	@#nvcc main.cu -gencode arch=compute_70,code=sm_70
	/opt/cuda/bin/nvcc -o main main.cu

clean:
	rm -rf *.out main ./build
