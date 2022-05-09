include sources.mk

CFLAGS =-std=c++11 $(INCLUDES:%=-I %)

CC = nvcc
LD= nvcc

OBJS = $(SOURCES:.cpp=.o)
all: *.cpp
	$(CC) $(CFLAGS) $(SOURCESLOCATION) -c
	$(LD) $(OBJS) -L /usr/local/cuda/lib64/ -lOpenCL -o cpukernel
.PHONY: clean
clean:
	rm -f $(OBJS) cpukernel $(IRRELEVANT)