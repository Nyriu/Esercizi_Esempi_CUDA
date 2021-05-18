all:
	@#nvcc -std=c++11 main.cu -gencode arch=compute_50,code=sm_50 -o main -Wno-deprecated-gpu-targets
	@/opt/cuda/bin/nvcc -std=c++11 main.cu -gencode arch=compute_70,code=sm_70 -o main

clean:
	rm -rf *.out main ./build
