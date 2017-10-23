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

// Returns the index of the first |array|
// element bigger than |element| or |size| if none exist.
// |array| is assumed to be sorted.
size_t UpperBound(int* array, size_t size, int element) {
  if (array[0] > element)
    return 0;
  size_t a = 0;
  size_t b = size;
  while (b - a > 1) {
    size_t c = (a + b) / 2;
    if (array[c] > element) {
      b = c;
    } else {
      a = c;
    }
  }
  return b;
}

void PrintArray(int* arr, size_t count, FILE* file) {
  size_t i;
  for (i = 0; i < count; i++) {
    fprintf(file, "%d ", arr[i]);
  }
  fprintf(file, "\n");
}

// Merge two arrays, one with |left_count| elements starting at
// |left|, another at |right| with |right_count| elements.
// Result is left in |buffer|.
void Merge(int* left,
           int* right,
           int* buffer,
           size_t left_count,
           size_t right_count) {
  size_t length = left_count + right_count;
  size_t left_cursor = 0;
  size_t right_cursor = 0;
  size_t buffer_cursor = 0;
  if (left_count == 0) {
    memcpy(buffer, right, right_count * sizeof(int));
    return;
  } else if (right_count == 0) {
    memcpy(buffer, left, left_count * sizeof(int));
    return;
  }
  while (buffer_cursor < length) {
    if (left_cursor == left_count ||
        (right_cursor < right_count &&
         right[right_cursor] < left[left_cursor])) {
      buffer[buffer_cursor] = right[right_cursor];
      ++right_cursor;
    } else if (right_cursor == length ||
               right[right_cursor] >= left[left_cursor]) {
      buffer[buffer_cursor] = left[left_cursor];
      ++left_cursor;
    }
    buffer_cursor++;
  }
}

// Merge two sequential arrays, one with |left_count| elements starting at
// zero, another at |left_count| with |right_count| elements.
// Buffer is assumed to have enough memory to store |left_count + right_count|.
void ParallelMerge(int* left,
                   int* buffer,
                   size_t left_count,
                   size_t right_count) {
  int* big;
  int* small;
  size_t big_count;
  size_t small_count;
  if (left_count < right_count) {
    big = left + left_count;
    big_count = right_count;
    small = left;
    small_count = left_count;
  } else {
    big = left;
    big_count = left_count;
    small = left + left_count;
    small_count = right_count;
  }
  size_t big_half = left_count / 2;
  int big_median = big[big_half];
  size_t small_upper = UpperBound(small, small_count, big_median);
  int* right_buffer = buffer + big_half + small_upper;
  // We need to preserve the stability of the sort.
  if (big_count == left_count) {
    #pragma omp task
    Merge(big, small, buffer, big_half, small_upper);
    #pragma omp task
    Merge(big + big_half, small + small_upper, right_buffer,
          big_count - big_half, small_count - small_upper);
  } else {
    #pragma omp task
    Merge(small, big, buffer, small_upper, big_half);
    #pragma omp task
    Merge(small + small_upper, big + big_half, right_buffer,
          small_count - small_upper, big_count - big_half);
  }
  #pragma omp taskwait
  memcpy(left, buffer, (right_count + left_count) * sizeof(int));
}

void RandomFill(int* arr, size_t count) {
  size_t i;
  for (i = 0; i < count; i++) {
    arr[i] = rand() % 16;
  }
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
    ParallelMerge(arr, buffer, middle, count - middle);
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
  free(buffer);
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
  fclose(data);
  fprintf(stats, "n: %lu, m: %lu, P: %d, t: %lf\n", length, chunk_size, threads,
          time_spent_merge);
  fclose(stats);

  assert(CheckSorted(array_merge, length));
  assert(!CheckSorted(array_qsort, length));

  double begin = omp_get_wtime();
  qsort(array_qsort, length, sizeof(int), sort_compare);
  double time_spent_qsort = omp_get_wtime() - begin;

  printf("Merge: %lf\nqsort: %lf\n", time_spent_merge, time_spent_qsort);

  assert(CheckSorted(array_merge, length));
  assert(CheckSorted(array_qsort, length));

  free(array_merge);
  free(array_qsort);
}
