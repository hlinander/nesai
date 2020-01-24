#undef OPENGL
#include "brain.h"
#include "model.h"

#include <stdlib.h>
#include <iostream>
#include <dlfcn.h>

#ifdef LUA_WITH_VERSION
	#include <lua5.3/lua.hpp>
#else
	#include <lua.hpp>
#endif

static uint8_t gp_bits = 0;

static torch::NoGradGuard guard;
static Model model{0.001};

static uint32_t rollouts = 1;
static bool enabled = false;
static bool headless = false;
static bool show_fps = false;
static uint32_t fps = 0;
static uint64_t next_fps = 0;
static uint32_t frame = 0;
static uint32_t save_frame = 0;
static const char* name = "noname";
static const char * expfile = nullptr;

static lua_State *L = nullptr;

void save_state();
void load_state();

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
static const uint32_t *nes_screen = nullptr;

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

static int brain_lua_readcpuint(lua_State *L)
{
	uint16_t addr = static_cast<uint16_t>(luaL_checknumber(L, 1));
	if(addr >= 0x2000)
	{
		std::cout << "LUA: Reading invalid address" << std::hex << addr;
		exit(1);
	}
	lua_pushnumber(L, static_cast<int8_t>(cpu_ram[addr & 0x7FF]));
	return 1;
}

static int brain_lua_load_state(lua_State *L)
{
	load_state();
	frame = save_frame;
	return 1;
}

static int brain_lua_save_state(lua_State *L)
{
	save_state();
	save_frame = frame;
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
	srand(time(0));
	const char *script = getenv("BE");

	torch::set_num_threads(1);

	if(nullptr == script)
	{
		enabled = false;
	}
	else
	{
		if(nullptr != (name = getenv("MODEL")))
		{
			if(!model.load_file(name))
			{
				std::cout << "NOO. PANTS" << std::endl;
				exit(1);
			}
			std::cout << "This one is for Hampus!" << std::endl;
		}
		else
		{
			std::cout << "WARNING!! Running without name" << std::endl;
		}

		expfile = getenv("EXPFILE");
		const char *ro = getenv("ROLLOUTS");
		if(nullptr != ro)
		{
			rollouts = static_cast<uint32_t>(atoi(ro));
		}

		enabled = true;
		if(nullptr == (L = luaL_newstate()))
		{
			std::cout << "Failed to initialize LUA" << std::endl;
			exit(1);
		}
		luaL_openlibs(L);

		printf("script %s\n", script);
		int status = luaL_dofile(L, script);

		if(status)
		{
			std::cout << "Error loading script (" << status << "): " << lua_tostring(L, -1);
			exit(1);
		}

		lua_pushcfunction(L, brain_lua_readcpu);
		lua_setglobal(L, "read_cpu");
		lua_pushcfunction(L, brain_lua_readcpuint);
		lua_setglobal(L, "read_int_cpu");
		lua_pushcfunction(L, brain_lua_log);
		lua_setglobal(L, "log");
		lua_pushcfunction(L, brain_lua_load_state);
		lua_setglobal(L, "load_state");
		lua_pushcfunction(L, brain_lua_save_state);
		lua_setglobal(L, "save_state");
		model.net->eval();
	}

	headless = bool_env("HL");
	show_fps = bool_env("FPS");
}

void brain_begin_rollout()
{
	lua_getglobal(L, "brain_begin_rollout");
	if(0 != lua_pcall(L, 0, 0, 0))
	{
		std::cout << "LUA: Error running 'brain_begin_rollout': " << lua_tostring(L, -1) << std::endl;
		exit(1);
	}
	fps = 0;
	next_fps = 0;
	frame = 0;
}

uint32_t brain_num_rollouts()
{
	return rollouts;	
}

bool brain_enabled()
{
	return enabled;
}

bool brain_headless()
{
	return enabled && headless;
}

static bool get_reward(uint32_t frame, float &ret)
{
	//
	//j Check if we are good...?
	//
	lua_getglobal(L, "brain_get_reward");
	lua_pushnumber(L, frame);
	if(0 != lua_pcall(L, 1, 1, 0))
	{
		std::cout << "LUA: Error running 'brain_get_reward': " << lua_tostring(L, -1) << std::endl;
		exit(1);
	}
	if(lua_isnil(L, -1))
	{
		lua_pop(L, 1);
		return false;
	}
	if(!lua_isnumber(L, -1))
	{
		std::cout << "LUA: 'brain_validate_frame' not returning an number" << std::endl;
		exit(1);	
	}
	
	ret = static_cast<float>(lua_tonumber(L, -1));
	// std::cout << ret << std::endl;
	lua_pop(L, 1);
	return true;
}

static int override_input(uint32_t frame)
{
	lua_getglobal(L, "brain_override_input");
	lua_pushnumber(L, frame);
	if(0 != lua_pcall(L, 1, 1, 0))
	{
		std::cout << "LUA: Error running 'brain_override_input': " << lua_tostring(L, -1) << std::endl;
		exit(1);
	}
	if(!lua_isnumber(L, -1))
	{
		std::cout << "LUA: 'brain_override_input' not returning an bool" << std::endl;
		exit(1);	
	}
	
	int ret = static_cast<int>(lua_tonumber(L, -1));
	lua_pop(L, 1);
	return ret;
}
static bool validate_frame(uint32_t frame)
{
	//
	// Check if we are good...?
	//
	lua_getglobal(L, "brain_validate_frame");
	lua_pushnumber(L, frame);
	if(0 != lua_pcall(L, 1, 1, 0))
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

void brain_bind_cpu_mem(const uint8_t *ram, const uint32_t *screen)
{
	cpu_ram = ram;
	nes_screen = screen;
	brain_begin_rollout();
}

bool brain_on_frame(float *frame_reward)
{
	if(!brain_enabled())
	{
		return true;
	}

	if(!cpu_ram)
	{
		std::cout << "RAM NOT BOUND!!!" << std::endl;
		exit(1);
	}

	if(!validate_frame(frame))
	{
		//
		// Last frame
		//
		std::cout << "I am done!" << std::endl;
		std::string out;
		if(expfile)
		{
			out = expfile;
		}
		else
		{
			out = name;
			out.append(".experience");
		}
		if(out.empty())
		{
			std::cout << "NOT SAVING EXP" << std::endl;
			return false;
		}
		model.save_file(out);
		std::cout << "I should exit now..." << std::endl;
		return false;
	}
	float reward = 0;
	bool save_frame = get_reward(frame, reward);
	*frame_reward = reward;

	StateType s;
	for(size_t i = 0; i < RAM_SIZE; ++i) {
		s[i] = static_cast<float>(cpu_ram[i]) / 255.0 - 0.5f;
	}
	for(size_t i = 0; i < SCREEN_PIXELS; ++i) {
		s[i*3 + RAM_SIZE] = static_cast<float>((nes_screen[i] >> 16) & 255) / 255.0 - 0.5f;
		s[i*3 + RAM_SIZE + 1] = static_cast<float>((nes_screen[i] >> 8) & 255) / 255.0 - 0.5f;
		s[i*3 + RAM_SIZE + 2] = static_cast<float>((nes_screen[i]) & 255) / 255.0 - 0.5f;
	}
	ActionType a = model.get_action(s);
	if(save_frame)
	{
		// std::cout << reward << std::endl;
		model.record_action(s, a, reward);
	}

	int override = override_input(frame);
	if(-1 == override)
	{
		size_t A = static_cast<size_t>(Action::A);
		size_t B = static_cast<size_t>(Action::B);
		size_t UP = static_cast<size_t>(Action::UP);
		size_t UP_A = static_cast<size_t>(Action::UP_A);
		size_t UP_B = static_cast<size_t>(Action::UP_B);
		size_t DOWN = static_cast<size_t>(Action::DOWN);
		size_t DOWN_A = static_cast<size_t>(Action::DOWN_A);
		size_t DOWN_B = static_cast<size_t>(Action::DOWN_B);
		size_t LEFT = static_cast<size_t>(Action::LEFT);
		size_t LEFT_A = static_cast<size_t>(Action::LEFT_A);
		size_t LEFT_B = static_cast<size_t>(Action::LEFT_B);
		size_t RIGHT = static_cast<size_t>(Action::RIGHT);
		size_t RIGHT_A = static_cast<size_t>(Action::RIGHT_A);
		size_t RIGHT_B = static_cast<size_t>(Action::RIGHT_B);

		gp_bits = 0;
		gp_bits |= a[A] << 0;
		gp_bits |= a[UP_A] << 0;
		gp_bits |= a[DOWN_A] << 0;
		gp_bits |= a[LEFT_A] << 0;
		gp_bits |= a[RIGHT_A] << 0;
		gp_bits |= a[B] << 1;
		gp_bits |= a[UP_B] << 1;
		gp_bits |= a[DOWN_B] << 1;
		gp_bits |= a[LEFT_B] << 1;
		gp_bits |= a[RIGHT_B] << 1;
		gp_bits |= a[UP] << 4;
		gp_bits |= a[UP_A] << 4;
		gp_bits |= a[UP_B] << 4;
		gp_bits |= a[DOWN] << 5;
		gp_bits |= a[DOWN_A] << 5;
		gp_bits |= a[DOWN_B] << 5;
		gp_bits |= a[LEFT] << 6;
		gp_bits |= a[LEFT_A] << 6;
		gp_bits |= a[LEFT_B] << 6;
		gp_bits |= a[RIGHT] << 7;
		gp_bits |= a[RIGHT_A] << 7;
		gp_bits |= a[RIGHT_B] << 7;
	}
	else
	{
		gp_bits = static_cast<uint8_t>(override);
	}
	++fps;
	auto now = get_ms();
	if(next_fps < now)
	{
		std::cout << "FPS: " << fps << std::endl;
		fps = 0;
		next_fps = now + 100;
	}
	++frame;
	return true;
}

//
// LUA-JIT bindings
//
static int luajit_brain_enabled(lua_State *L)
{
	lua_pushboolean(L, brain_enabled());
	return 1;
}

static int luajit_brain_headless(lua_State *L)
{
	lua_pushboolean(L, brain_headless());
	return 1;
}

static int luajit_brain_controller_bits(lua_State *L)
{
	lua_pushinteger(L, brain_controller_bits());
	return 1;
}

// static int luajit_brain_on_frame(lua_State *L)
// {
// 	lua_pushboolean(L, brain_on_frame());
// 	return 1;
// }

extern "C" int luaopen_brain_luajit(lua_State *L)
{
	static const luaL_Reg functions[] = {
		{ "enabled", luajit_brain_enabled },
		{ "headless", luajit_brain_headless },
		{ "controller_bits", luajit_brain_controller_bits },
		// { "on_frame", luajit_brain_on_frame },
		{ nullptr, nullptr }
	};
	printf("start\n");
	brain_init();
	printf("start2\n");
	lua_newtable(L);
	printf("start3\n");
	luaL_setfuncs(L, functions, 0);

	void *hq = dlopen("./libhqnes.so", RTLD_NOW);
	if(hq)
	{
		printf("Found libhq! Doing hacks.\n");
		typedef const uint8_t * (*get_mem)(void);
		get_mem gm = (get_mem)dlsym(hq, "hq_get_cpu_mem");
		if(gm)
		{
			printf("...oh and found function too!\n");
			cpu_ram = gm();
		}
	}

	return 1;
}




