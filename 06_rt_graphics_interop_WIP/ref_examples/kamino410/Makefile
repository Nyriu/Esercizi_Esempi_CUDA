## all:
## 	@nvcc -std=c++11 main.cu -gencode arch=compute_50,code=sm_50 -o main -Wno-deprecated-gpu-targets -lglut -lGL -lGLU
## 
## clean:
## 	rm -rf *.out main ./build


NVCCFLAGS := -g -O3 --use_fast_math -std=c++11 -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets

LIBS := -lGL -lglfw -lGLEW
#LIBS := -lglut -lGL -lGLU -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,%,$(SRCS))

all: $(BINS)
		
%: %.cu
	nvcc $(NVCCFLAGS) $(LIBS) $< -o $@

clean:
	rm -rf *.out main ./build
