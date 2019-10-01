INCLUDES  = -I/usr/include/torch/csrc/api/include
INCLUDES += -Icereal-1.2.2/include
CPPFLAGS  = -std=c++17 -O3 -Wall -fPIC -g -ggdb
LIBS = -lluajit-5.1 -lcuda -ltorch -lc10

PHONY: clean
PHONY: all

all: overmind libbrain.so

clean:
	rm -fr *.o *.so

overmind: overmind.o
	g++ $< $(LIBS) -o $@

libbrain.so: brain.o
	g++ $< -shared $(LIBS) -o $@

overmind.o: Makefile overmind.cpp model.h brain.h bm.h
	g++ overmind.cpp $(INCLUDES) $(CPPFLAGS) -c

brain.o: Makefile brain.cpp model.h brain.h bm.h
	g++ brain.cpp $(INCLUDES) $(CPPFLAGS) -c

