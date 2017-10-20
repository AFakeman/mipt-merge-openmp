CFLAGS = -Wall -Werror -fopenmp -g

main: main.c
	gcc-7 main.c -o main $(CFLAGS)
