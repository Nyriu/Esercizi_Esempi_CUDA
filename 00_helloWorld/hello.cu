#include "stdio.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Prima di cuda");
    cuda_hello<<<1,1>>>(); 
    printf("Dopo di cuda");
    return 0;
}
