LIB_QUICKNES = ../quicknes/libquicknes.a

ifeq ($(lib), yes)

TARGET = ../libhqnes.so

LINKFLAGS = -L ../quicknes/ -lSDL2_ttf -lquicknes -ldl `pkg-config --libs sdl2 luajit -shared`
CXXFLAGS = -std=c++11 -I ../quicknes -O3 -Wno-multichar -fno-exceptions -fpermissive -fPIC \
		   `pkg-config --cflags sdl2 luajit` -g -ggdb

SOURCES = \
	../src/hqn.cpp \
	../src/hqn_util.cpp \
	../src/hqn_lua.cpp \
	../src/hqn_lua_emu.cpp \
	../src/hqn_lua_gui.cpp \
	../src/hqn_lua_input.cpp \
	../src/hqn_lua_joypad.cpp \
	../src/hqn_lua_mainmemory.cpp \
	../src/hqn_lua_savestate.cpp \
	../src/hqn_gui_controller.cpp \
	../src/hqn_surface.cpp
#   ../src/hqn_main.cpp    <-- Don't include this because it's not necessary for shared build

else

TARGET = ../hqn_quicknes

LINKFLAGS = -L ../quicknes/ -lSDL2_ttf -llua5.3 -lcuda -ltorch -lc10 -lquicknes -ldl `pkg-config --libs sdl2`

CXXFLAGS = -std=c++11 -I ../quicknes -O3 -g -ggdb -Wno-multichar -fpermissive -fPIC \
		   `pkg-config --cflags sdl2`

CXXFLAGS += -I../../ai
CXXFLAGS += -I../../ai/cereal-1.2.2/include
CXXFLAGS += -I/usr/include/torch/csrc/api/include

SOURCES = \
	../src/hqn.cpp \
	../src/hqn_util.cpp \
	../src/hqn_gui_controller.cpp \
	../src/hqn_surface.cpp \
	../src/hqn_native.cpp \
	../../ai/brain.cpp

endif

OBJECTS = $(SOURCES:.cpp=.o) \

%.o: %.cpp
	echo $(CXXFLAGS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

# Build as shared library
$(TARGET): $(OBJECTS) $(LIB_QUICKNES)
	$(CXX) -o $@ $(OBJECTS) $(LINKFLAGS) -fPIC

$(LIB_QUICKNES):
	$(MAKE) -C ../quicknes/linux

clean:
	$(RM) $(OBJECTS)
	$(RM) $(LIB_QUICKNES)
	$(RM) $(TARGET)

