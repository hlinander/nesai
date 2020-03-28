INCLUDES  = -I/pytorch/libtorch/include
INCLUDES += -I/pytorch/libtorch/include/torch/csrc/api/include/
INCLUDES += -Icereal-1.2.2/include
CPPFLAGS  = -std=c++17 -O3 -Wall -fPIC -g -ggdb
LIBS = -llua5.3 -ltorch -lc10 -lstdc++fs -lcurses -lfuzzy

ifeq ($(hampus), off)
	CPPFLAGS += -DNO_HAMPUS
endif

ifneq ($(GPU),0)
	LIBS += -lcuda
endif

ifneq ($(findstring -ARCH,$(shell uname -a)),-ARCH)
	CPPFLAGS += -DLUA_WITH_VERSION
else
	CPPFLAGS += -Wno-deprecated-declarations
endif

.PHONY: all clean

all: overmind libbrain.so

overmind: overmind.o replay.o reward.o
	g++ $^ $(LIBS) -o $@

libbrain.so: brain.o lodepng.o
	g++ $< -shared $(LIBS) -o $@

lodepng.o: lodepng.h
	g++ lodepng.cpp $(INCLUDES) $(CPPFLAGS) -c

overmind.o: Makefile overmind.cpp model.h brain.h bm.h reward.h
	g++ overmind.cpp $(INCLUDES) $(CPPFLAGS) -c

replay.o: Makefile replay.cpp replay.h model.h
	g++ replay.cpp $(INCLUDES) $(CPPFLAGS) -c

reward.o: Makefile reward.cpp reward.h model.h
	g++ reward.cpp $(INCLUDES) $(CPPFLAGS) -c

brain.o: Makefile brain.cpp model.h brain.h bm.h
	g++ brain.cpp $(INCLUDES) $(CPPFLAGS) -c

clean:
	rm -fr overmind libbrain.so *.o *.so
