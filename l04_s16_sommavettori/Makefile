all:
	#nvcc -std=c++11 main.cu -o main
	nvcc -o main main.cu
	#nvcc main.cu -gencode arch=compute_50,code=sm_50
	nvcc main.cu -gencode arch=compute_50,code=sm_50 -o main

clean:
	rm -rf *.out main ./build
