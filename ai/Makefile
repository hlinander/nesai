INCLUDES  = -I/pytorch/libtorch/include
INCLUDES += -I/pytorch/libtorch/include/torch/csrc/api/include/
INCLUDES += -Icereal-1.2.2/include
CPPFLAGS  = -std=c++17 -O0 -Wall -fPIC -g -ggdb 
LIBS = -llua5.3 -lcuda -ltorch -lc10 -lstdc++fs

ifeq ($(hampus), off)
	CPPFLAGS += -DNO_HAMPUS
endif

ifneq ($(findstring -ARCH,$(shell uname -a)),-ARCH)
	CPPFLAGS += -DLUA_WITH_VERSION
else
	CPPFLAGS += -Wno-deprecated-declarations
endif

.PHONY: all clean

all: overmind libbrain.so

overmind: overmind.o
	g++ $< $(LIBS) -o $@

libbrain.so: brain.o
	g++ $< -shared $(LIBS) -o $@

overmind.o: Makefile overmind.cpp model.h brain.h bm.h
	g++ overmind.cpp $(INCLUDES) $(CPPFLAGS) -c

brain.o: Makefile brain.cpp model.h brain.h bm.h
	g++ brain.cpp $(INCLUDES) $(CPPFLAGS) -c

clean:
	rm -fr overmind libbrain.so *.o *.so
