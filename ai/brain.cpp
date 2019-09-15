#undef OPENGL
#include "brain.h"
#include "model.h"

#include <stdlib.h>
#include <iostream>

extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

static uint8_t gp_bits = 0;

static Model model{0.001};

static bool enabled = false;
static bool headless = false;
static bool show_fps = false;
static uint32_t fps = 0;
static uint64_t next_fps = 0;

static lua_State *L = nullptr;

static bool bool_env(const char *env)
{
	const char * const s = getenv(env);
	if(!s)
	{
		return false;
	}
	return (0 != strcmp(s, "0"));
}

static uint64_t get_ms() 
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);

	uint64_t ms  = ts.tv_nsec / 1000000;
	ms += ts.tv_sec * 1000;
	return ms;
}

static const uint8_t *cpu_ram = nullptr;

static int brain_lua_readcpu(lua_State *L)
{
	uint16_t addr = static_cast<uint16_t>(luaL_checknumber(L, 1));
	if(addr >= 0x2000)
	{
		std::cout << "LUA: Reading invalid address" << std::hex << addr;
		exit(1);
	}
	lua_pushnumber(L, cpu_ram[addr & 0x7FF]);
	return 1;
}

static int brain_lua_log(lua_State *L)
{
	const char *p = luaL_checkstring(L, 1);
	if(nullptr == p)
	{
		std::cout << "LUA: Missing log argument" << std::endl;
		exit(1);		
	}
	std::cout << "[LUA]: " << p << std::endl;
	return 0;
}

void brain_init()
{
	const char *script = getenv("BE");

	if(nullptr == script)
	{
		enabled = false;
	}
	else
	{
		enabled = true;
		if(nullptr == (L = luaL_newstate()))
		{
			std::cout << "Failed to initialize LUA" << std::endl;
			exit(1);
		}
		luaL_openlibs(L);

		int status = luaL_dofile(L, script);

		if(status)
		{
			std::cout << "Error loading script (" << status << "): " << lua_tostring(L, -1);
			exit(1);
		}

		lua_pushcfunction(L, brain_lua_readcpu);
		lua_setglobal(L, "read_cpu");
		lua_pushcfunction(L, brain_lua_log);
		lua_setglobal(L, "log");
	}

	headless = bool_env("HL");
	show_fps = bool_env("FPS");
}

bool brain_enabled()
{
	return enabled;
}

bool brain_headless()
{
	return enabled && headless;
}

static bool validate_frame()
{
	//
	// Check if we are good...?
	//
	lua_getglobal(L, "brain_validate_frame");
	if(0 != lua_pcall(L, 0, 1, 0))
	{
		std::cout << "LUA: Error running 'brain_validate_frame': " << lua_tostring(L, -1) << std::endl;
		exit(1);
	}
	if(!lua_isboolean(L, -1))
	{
		std::cout << "LUA: 'brain_validate_frame' not returning an bool" << std::endl;
		exit(1);	
	}
	
	bool ret = static_cast<bool>(lua_toboolean(L, -1));
	lua_pop(L, 1);
	return ret;
}

uint8_t brain_controller_bits()
{
	return gp_bits;
}

bool brain_on_frame(const uint8_t *ram, size_t n)
{
	if(!brain_enabled())
	{
		return true;
	}

	cpu_ram = ram;

	if(!validate_frame())
	{
		return false;
	}

	if(n != STATE_SIZE)
	{
		std::cout << "BAD RAM SIZE (on frame): " << n << " vs. " << STATE_SIZE << std::endl;
		exit(1);
	}

	StateType s;
	for(size_t i = 0; i < STATE_SIZE; ++i) {
		s[i] = static_cast<float>(ram[i]) / 255.0;
	}

	ActionType a = model.get_action(s);
	gp_bits = 0;
	gp_bits |= a[static_cast<size_t>(Action::A)] << 0;
	gp_bits |= a[static_cast<size_t>(Action::B)] << 1;
	gp_bits |= a[static_cast<size_t>(Action::SELECT)] << 2;
	gp_bits |= a[static_cast<size_t>(Action::START)] << 3;
	gp_bits |= a[static_cast<size_t>(Action::UP)] << 4;
	gp_bits |= a[static_cast<size_t>(Action::DOWN)] << 5;
	gp_bits |= a[static_cast<size_t>(Action::LEFT)] << 6;
	gp_bits |= a[static_cast<size_t>(Action::RIGHT)] << 7;

	++fps;
	auto now = get_ms();

	if(next_fps < now)
	{
		std::cout << "FPS: " << fps << std::endl;
		fps = 0;
		next_fps = now + 1000;
	}
	return true;
}
