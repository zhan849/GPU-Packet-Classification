CXX = nvcc
CXX_FLAGS = -Wall -g -o

out: main.o setmerge.o
	$(CXX) $(CXX_FLAGS) out main.o setmerge.o

main.o: main.cu setmerge.h
	$(CXX) $(CXX_FLAGS) main.cu

setmerge.o: setmerge.cu setmerge.h
	$(CXX) $(CXX_FLAGS) setmerge.cu

clean:
	rm -f *.o out