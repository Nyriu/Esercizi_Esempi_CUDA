#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <string.h>

int main() {

#pragma omp parallel num_threads(10)
  {
    printf("Codice parallelo eseguito da %d thread\n", omp_get_thread_num());
    if (omp_get_thread_num() == 2) {
      printf("Il thread %d fa cose diverse\n", omp_get_thread_num());
    }
  } // fine parallelo

  exit(0);
}

