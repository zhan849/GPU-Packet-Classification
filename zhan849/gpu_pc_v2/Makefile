CXX = nvcc

CXX_FLAGS = -lpthread

default: out
all: out

test: out.o
	$(CXX) $(CXX_FLAGS) -o out gpu_pc_v2_func.o gpu_pc_v2_1.o

gpu_pc_v2_1.o: gpu_pc_v2_func.cu gpu_pc_v2_1.cu
	$(CXX) $(CXX_FLAGS) gpu_pc_v2_func.cu gpu_pc_v2_1.cu

clean:
	/bin/rm -f test gpu_pc_v2_func.o gpu_pc_v2_1.o
	
squeaky:
	rm *.o test
	rm *~