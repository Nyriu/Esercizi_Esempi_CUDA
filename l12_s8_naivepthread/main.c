#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>

int var;

void print_msg(void *ptr) {
  int x = *((int*)ptr); // cast and deref

  // pthread_t e' unsigned long
  printf("Thread %lu : x=%d\n", pthread_self(), x);

  var = (x>5) ? (var+100) : (var*3);
}

int main() {
  pthread_t thread_1, thread_2;
  int val1 = 8;
  int val2 = 3;

  var = 2;

  if (pthread_create(&thread_1, NULL, (void*)&print_msg, (void*)&val1) != 0) {
    perror("Errore creazione primo thread\n");
    exit(1);
  }

  if (pthread_create(&thread_2, NULL, (void*)&print_msg, (void*)&val2) != 0) {
    perror("Errore creazione secondo thread\n");
    exit(2);
  }

  pthread_join(thread_1, NULL);
  pthread_join(thread_2, NULL);
  printf("Finale=%d\n", var); // not deterministic!
  exit(0);
}

