CFLAGS = -Wall -Werror -fopenmp -g
CC = gcc-7

main: main.c
	$(CC) main.c -o main $(CFLAGS)
