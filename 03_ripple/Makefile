NVCCFLAGS := -g -O3 --use_fast_math -std=c++11 -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets
LIBS := -lglut -lGL -lGLU
SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,%,$(SRCS))

all: $(BINS)
		
%: %.cu
	nvcc $(NVCCFLAGS) $(LIBS) $< -o $@

clean:
	rm -rf *.out main ./build
