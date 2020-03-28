#undef OPENGL
#include "brain.h"
#include "model.h"
#include "lodepng.h"

#include <stdlib.h>
#include <iostream>
#include <dlfcn.h>
#include <fuzzy.h>

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

static uint32_t brain_screen[SCREEN_PIXELS];

static std::vector<std::unique_ptr<char []>> frame_history;

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
	memset(brain_screen, 0, sizeof(brain_screen));

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
		model.value_net->eval();
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

void brain_bind_nes(const uint8_t *ram, const uint32_t *screen)
{
	cpu_ram = ram;
	nes_screen = screen;
	brain_begin_rollout();
}

bool brain_on_frame(float *frame_reward, int *action_idx)
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
	bool save_it = get_reward(frame, reward);
	save_it = true;

	*frame_reward = reward;

	StateType s;
	for(size_t i = 0; i < RAM_SIZE; ++i) {
		s[i] = static_cast<float>(cpu_ram[i]) / 255.0 - 0.5f;
	}

	for(int y = 0; y < SCREEN_H; ++y)
	{
		for(int x = 0; x < SCREEN_W; ++x)
		{
			// uint32_t hash = 0xFFFFFFFF;
			uint32_t r = 0;
			uint32_t g = 0;
			uint32_t b = 0;

			const int block_w = (256 / SCREEN_W);
			const int block_h = (240 / SCREEN_H);

			for(int i = 0; i < block_w; ++i)
			{
				for(int j = 0; j < block_h; ++j)
				{
					uint32_t idx = (y * block_w + j) * 256 + x * block_w + i;
					r += nes_screen[idx] >> 16 & 0xff;
					g += nes_screen[idx] >> 8 & 0xff;
					b += nes_screen[idx] & 0xff;
				}
			}
			brain_screen[y * SCREEN_W + x] = 0xFF000000 | ((b >> 6) << 16) | ((g >> 6) << 8) | (r >> 6);
		}
	}

	std::unique_ptr<char []> hash(new char [FUZZY_MAX_RESULT]);
	fuzzy_hash_buf(reinterpret_cast<const uint8_t *>(brain_screen),
		sizeof(brain_screen), hash.get());

	if(frame_history.size())
	{
		int most_similar = 0;
		int at_frame = 0;
		int sim_frame = 0;
		for(const auto &it : frame_history)
		{
			int score = fuzzy_compare(it.get(), hash.get());
			if(score > most_similar)
			{
				most_similar = score;
				sim_frame = at_frame;
			}
			++at_frame;
		}

// #define DEBUG_FRAMES
#ifdef DEBUG_FRAMES
		std::cout << "Frame " << frame << " looks like " << sim_frame << ", score: " << most_similar << std::endl;
		std::vector<uint32_t> png;
		png.resize(SCREEN_PIXELS * 2);

		const uint32_t full_w = (SCREEN_W * 2);

		for(uint32_t y = 0; y < SCREEN_H; ++y)
		{
			memcpy(png.data() + (y * full_w),
				brain_screen + (y * SCREEN_W),
				SCREEN_W * 4);

			if(!most_similar)
			{
				for(uint32_t x = 0; x < SCREEN_W; ++x)
				{
					png[SCREEN_W + (y * full_w) + x] = 0xFF0000FF;
				}
			}
			else
			{
				const auto &sim = model.states[sim_frame];
				for(uint32_t x = 0; x < SCREEN_W; ++x)
				{
					uint32_t idx = ((y * SCREEN_W) + x) * 3;
					uint32_t rgb = 0xFF000000;
					rgb |= (static_cast<uint32_t>((sim[RAM_SIZE + idx + 0] + 0.5f) * 255.0f) << 16);
					rgb |= (static_cast<uint32_t>((sim[RAM_SIZE + idx + 1] + 0.5f) * 255.0f) << 8);
					rgb |= (static_cast<uint32_t>((sim[RAM_SIZE + idx + 2] + 0.5f) * 255.0f) << 0);
					png[SCREEN_W + (y * full_w) + x] = rgb;
				}
			}
		}

		std::stringstream kek;
		kek << "test-" << frame << "-" << sim_frame << "-" << most_similar << ".png";
		lodepng_encode32_file(kek.str().c_str(), (const uint8_t *)png.data(), SCREEN_W * 2, SCREEN_H);
#endif
	}

	frame_history.emplace_back(std::move(hash));

	// TODO - Do this garbage in the pass above that does literarly the exakt same thing.......
	for(size_t i = 0; i < SCREEN_PIXELS; ++i) {
		s[i*3 + RAM_SIZE] = static_cast<float>((brain_screen[i] >> 16) & 255) / 255.0 - 0.5f;
		s[i*3 + RAM_SIZE + 1] = static_cast<float>((brain_screen[i] >> 8) & 255) / 255.0 - 0.5f;
		s[i*3 + RAM_SIZE + 2] = static_cast<float>((brain_screen[i]) & 255) / 255.0 - 0.5f;
	}
	ActionType a = model.get_action(s);
	for(uint32_t i = 0; i < ACTION_SIZE; ++i) {
		if(a[i] == 1)
		{
			*action_idx = i;
		}
	}

	float v = model.get_value(s);
	if(save_it)
	{
		// std::cout << reward << std::endl;
		// std::cout << v << std::endl;
		model.record_action(s, a, reward, v);
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

		// gp_bits = 1 << 7;
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
		next_fps = now + 1000;
	}
	++frame;
	return true;
}

const uint32_t *brain_get_screen(uint32_t &w, uint32_t &h)
{
	w = static_cast<uint32_t>(SCREEN_W);
	h = static_cast<uint32_t>(SCREEN_H);

	return brain_screen;
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




