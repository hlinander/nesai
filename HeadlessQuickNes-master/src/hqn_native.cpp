#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

std::string read_env(const char *name)
{
	std::string s;
	DWORD n = GetEnvironmentVariableA("HL", NULL, 0);
	s.resize(n);
	GetEnvironmentVariableA("HL", s.data(), s.size());
	return s;
}

#else
#include <unistd.h>

std::string read_env(const char *name)
{
	std::string s;
	const char *p = getenv(name);
	if(p)
	{
		s = p;
	}
	return s;
}
#endif

static const bool brain_headless = !read_env("HL").empty();
static const std::string brain_script = read_env("BE");

static bool brain_enabled()
{
	return !brain_script.empty();
}

static void ngui_setscale(int scale = 1)
{
	STATE_GUI(state, gui);
	gui->setScale(scale);
}

static bool ngui_enable()
{
    HQN_STATE(state);
 
    if(!SDL_WasInit(SDL_INIT_VIDEO))
    {
        if(SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
        {
            return false;
        }
    }

    GET_GUI(state, gui);
    if(!gui)
    {
        hqn::GUIController *controller = nullptr;

        if(controller = hqn::GUIController::create(*state))
        {
            state->setListener(controller);
            controller->setCloseOperation(hqn::GUIController::CLOSE_DELETE);
        }
        else
        {
            return false;
        }
    }
    return true;
}

static void nemu_setframerate(int fps)
{
	HQN_STATE(state);
	state->setFramerate(fps);
}

static bool nemu_loadrom(const char *romname)
{
	HQN_STATE(state);
	const char *err = state->loadROM(romname);
	if(err)
	{
		std::cout << "Error when loading ROM " << romname << ": " << err << std::endl;
		return false;
	}
	return true;
}

static void njoypad_set(int pad, uint8_t bits)
{
	HQN_STATE(state);
	state->joypad[pad] = bits;
}

static void nemu_frameadvance()
{
	HQN_STATE(state);
	state->advanceFrame();
}

struct gamepad_binding
{
	int sdl_scancode;
	uint8_t bit;
};

static int run_human_mode()
{
	gamepad_binding bindings[] = 
	{
		// A
		{ SDL_SCANCODE_X, 0x01 },
		// B
		{ SDL_SCANCODE_Z, 0x02 },
		// Select
		{ SDL_SCANCODE_A, 0x04 },
		// Start
		{ SDL_SCANCODE_S, 0x08 },
		// Up
		{ SDL_SCANCODE_UP, 0x10 },
		// Down
		{ SDL_SCANCODE_DOWN, 0x20 },
		// Left
		{ SDL_SCANCODE_LEFT, 0x40 },
		// RIGHT
		{ SDL_SCANCODE_RIGHT, 0x80 },
	};

	for(;;)
	{
		uint8_t bits = 0;
		const Uint8 *kb = SDL_GetKeyboardState(NULL);
		
		for(size_t i = 0; i < (sizeof(bindings) / sizeof(bindings[0]); ++i)
		{
			const auto &b = bindings[i];
			if(kb[b.sdl_scancode])
			{
				bits |= b.bit;
			}
		}

		njoypad_set(0, bits);
		nemu_frameadvance();
	}

	return 0;
}

static int run_brain_mode()
{
	
}

int main(int argc, const char *argv[])
{
	if(argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <rom>" << std::endl;
		return 1;
	}

	if(!brain_headless)
	{
		if(!ngui_enable())
		{
			std::cout << "Unable to enable GUI" << std::endl;
			return 1;
		}
		ngui_setscale(2);
	}

	nemu_setframerate(0);

	if(!nemu_loadrom(argv[1]))
	{
		std::cout << "No rom :(" << std::endl;
		return 1;
	}

	if(brain_enabled())
	{
		return run_brain_mode();
	}

	return run_human_mode();
}

