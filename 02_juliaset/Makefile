all:
	@nvcc -std=c++11 main.cu -gencode arch=compute_50,code=sm_50 -o main -Wno-deprecated-gpu-targets

clean:
	rm -rf *.out main ./build
