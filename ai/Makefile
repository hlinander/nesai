INCLUDES  = -I/usr/include/torch/csrc/api/include
INCLUDES += -Icereal-1.2.2/include
CPPFLAGS  = -std=c++17 -O3 -Wall -fPIC
LIBS = -llua -lcuda -ltorch -lc10

PHONY: clean
PHONY: all

all: libbrain.so

clean:
	rm -fr *.o *.so

libbrain.so: brain.o
	g++ $< -shared $(LIBS) -o $@

brain.o: Makefile brain.cpp model.h brain.h
	g++ brain.cpp $(INCLUDES) $(CPPFLAGS) -c

