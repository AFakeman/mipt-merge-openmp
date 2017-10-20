#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

char* kStatsFile = "stats.txt";
char* kDataFile = "data.txt";

int sort_compare(const void* a, const void* b) {
  int a_val = *((const int*)a);
  int b_val = *((const int*)b);
  return a_val - b_val;
}

// Merge two sequential arrays, one with |left_count| elements starting at
// zero, another at |left_count| with |right_count| elements.
// Buffer is assumed to have enough memory to store |left_count + right_count|.
void Merge(int* left, int* buffer, size_t left_count, size_t right_count) {
  size_t length = left_count + right_count;
  size_t left_cursor = 0;
  size_t right_cursor = left_count;
  size_t buffer_cursor = 0;
  while (buffer_cursor < length) {
    if (left_cursor == left_count ||
        (right_cursor < length && left[right_cursor] < left[left_cursor])) {
      buffer[buffer_cursor] = left[right_cursor];
      ++right_cursor;
    } else if (right_cursor == length ||
               left[right_cursor] >= left[left_cursor]) {
      buffer[buffer_cursor] = left[left_cursor];
      ++left_cursor;
    }
    buffer_cursor++;
  }
  memcpy(left, buffer, sizeof(int) * length);
}

void RandomFill(int* arr, size_t count) {
  size_t i;
  for (i = 0; i < count; i++) {
    arr[i] = rand() % 16;
  }
}

void PrintArray(int* arr, size_t count, FILE* file) {
  size_t i;
  for (i = 0; i < count; i++) {
    fprintf(file, "%d ", arr[i]);
  }
  fprintf(file, "\n");
}

// Sort an array |arr| with length |count| recursively. If an array is
// not bigger than |min_count|, apply quick sort algorithm.
void RecursiveSort(int* arr, int* buffer, size_t count, size_t min_count) {
  if (count <= min_count) {
    qsort(arr, count, sizeof(int), sort_compare);
  } else {
    size_t middle = count / 2;
#pragma omp task
    RecursiveSort(arr, buffer, middle, min_count);
#pragma omp task
    RecursiveSort(arr + middle, buffer + middle, count - middle, min_count);
#pragma omp taskwait
    Merge(arr, buffer, middle, count - middle);
  }
}

int CheckSorted(int* arr, size_t count) {
  size_t i = 0;
  for (i = 1; i < count; i++) {
    if (arr[i - 1] > arr[i]) {
      return 0;
    }
  }
  return 1;
}

double MergeSort(int* array_merge,
                 size_t length,
                 size_t chunk_size,
                 int threads) {
  int* buffer = malloc(sizeof(int) * length);
  double begin = omp_get_wtime();
#pragma omp parallel num_threads(threads)
  {
#pragma omp single
    RecursiveSort(array_merge, buffer, length, chunk_size);
  }
  return omp_get_wtime() - begin;
}

int main(int argc, char* argv[]) {
  int* array_merge;
  int* array_qsort;
  size_t length;
  size_t chunk_size;
  int threads;
  FILE* stats = fopen(kStatsFile, "w");
  FILE* data = fopen(kDataFile, "w");

  srand(time(NULL));

  assert(argc == 4);
  sscanf(argv[1], "%lu", &length);
  sscanf(argv[2], "%lu", &chunk_size);
  sscanf(argv[3], "%d", &threads);

  array_merge = malloc(sizeof(int) * length);
  array_qsort = malloc(sizeof(int) * length);
  RandomFill(array_merge, length);
  memcpy(array_qsort, array_merge, sizeof(int) * length);
  PrintArray(array_merge, length, data);

  assert(!CheckSorted(array_merge, length));
  assert(!CheckSorted(array_qsort, length));

  double time_spent_merge = MergeSort(array_merge, length, chunk_size, threads);
  PrintArray(array_merge, length, data);
  fprintf(stats, "n: %lu, m: %lu, P: %d, t: %lf\n", length, chunk_size, threads,
          time_spent_merge);
  fclose(data);

  assert(CheckSorted(array_merge, length));
  assert(!CheckSorted(array_qsort, length));

  double begin = omp_get_wtime();
  qsort(array_qsort, length, sizeof(int), sort_compare);
  double time_spent_qsort = omp_get_wtime() - begin;

  printf("Merge: %lf\nqsort: %lf", time_spent_merge, time_spent_qsort);

  assert(CheckSorted(array_merge, length));
  assert(CheckSorted(array_qsort, length));

  free(array_merge);
  free(array_qsort);
}
