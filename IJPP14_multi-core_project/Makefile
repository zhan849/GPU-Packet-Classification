CXX = g++

CXX_FLAGS = -lpthread

default: all
all: test

test: main.o
	$(CXX) $(CXX_FLAGS) -lrt -o go func.o main.o

main.o: func.cpp main.cpp
	$(CXX) $(CXX_FLAGS) -lrt -c func.cpp main.cpp

clean:
	/bin/rm -f test func.o main.o
	
squeaky:
	rm *.o test
	rm *~