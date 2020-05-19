INCLUDES  = -I/pytorch/libtorch/include
INCLUDES += -I/pytorch/libtorch/include/torch/csrc/api/include/
INCLUDES += -Icereal-1.2.2/include
CPPFLAGS  = -std=c++17 -O0 -Wall -fPIC -ggdb
LIBS = -llua5.3 -ltorch -lc10 -lstdc++fs -lcurses

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

.PHONY: all clean python

all: overmind libai.a 
# libbrain.so

python: libai.a
	python3 setup.py bdist_wheel

overmind: model.o overmind.o replay.o reward.o tests.o
	g++ $^ $(LIBS) -o $@

libai.a: brain.o lodepng.o reward.o model.o
	ar -crs $@ $^

libbrain.so: brain.o lodepng.o model.o
	g++ $< -shared $(LIBS) -o $@

lodepng.o: lodepng.h
	g++ lodepng.cpp $(INCLUDES) $(CPPFLAGS) -c

overmind.o: Makefile overmind.cpp model.h brain.h bm.h reward.h rds.hpp
	g++ overmind.cpp $(INCLUDES) $(CPPFLAGS) -c

replay.o: Makefile replay.cpp replay.h model.h
	g++ replay.cpp $(INCLUDES) $(CPPFLAGS) -c

reward.o: Makefile reward.cpp reward.h model.h
	g++ reward.cpp $(INCLUDES) $(CPPFLAGS) -c

tests.o: tests.cpp reward.cpp reward.h
	g++ tests.cpp $(INCLUDES) $(CPPFLAGS) -c

brain.o: Makefile brain.cpp model.h brain.h bm.h
	g++ brain.cpp $(INCLUDES) $(CPPFLAGS) -c

model.o: Makefile model.cpp model.h brain.h bm.h reward.h
	g++ model.cpp $(INCLUDES) $(CPPFLAGS) -c


clean:
	rm -fr overmind libbrain.so *.o *.so
