#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>

int main() {

  int num_threads, sample_points_per_thread, npoints, sum=0;
  double rand_no_x, rand_no_y;
  unsigned int seed = 12345;

#pragma omp parallel \
  default(private) \
  shared(npoints) \
  reduction(+: sum) num_threads(8)
  {
    num_threads = omp_get_num_threads();
    sample_points_per_thread = npoints/num_threads;
    sum=0;
    for (int i=0; i<sample_points_per_thread; i++) {
      rand_no_x=(double)(rand_r(&seed))/(double)((2<<14)-1);
      rand_no_y=(double)(rand_r(&seed))/(double)((2<<14)-1);
      if (((rand_no_x - 0.5) * (rand_no_x - 0.5) +
            (rand_no_y - 0.5) * (rand_no_y - 0.5)) < 0.25)
        sum++;
    }
  }

  printf("sum=%d\n", sum);

  exit(0);
}

