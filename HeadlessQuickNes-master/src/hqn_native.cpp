#include "hqn.h"
#include "hqn_gui_controller.h"

#include <brain.h>

#include <string>
#include <iostream>
#include <SDL.h>
#include <SDL_video.h>
#include <SDL_keyboard.h>
#include <SDL_scancode.h>
#include <SDL_mouse.h>

using namespace hqn;

static HQNState hqn_state;

#define GET_GUI() hqn::GUIController *gui = static_cast<hqn::GUIController*>(hqn_state.getListener())

static void ngui_setscale(int scale = 1)
{
	GET_GUI();
	gui->setScale(scale);
}

static bool ngui_enable()
{
    if(!SDL_WasInit(SDL_INIT_VIDEO))
    {
        if(SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
        {
            return false;
        }
    }

    GET_GUI();
    if(!gui)
    {
        hqn::GUIController *controller = nullptr;

        if(controller = hqn::GUIController::create(hqn_state))
        {
            hqn_state.setListener(controller);
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
	hqn_state.setFramerate(fps);
}

static bool nemu_loadrom(const char *romname)
{
	const char *err = hqn_state.loadROM(romname);
	if(err)
	{
		std::cout << "Error when loading ROM " << romname << ": " << err << std::endl;
		return false;
	}
	return true;
}

static void njoypad_set(int pad, uint8_t bits)
{
	hqn_state.joypad[pad] = bits;
}

static void nemu_frameadvance()
{
	hqn_state.advanceFrame();
}

static bool ngui_isenabled()
{
    GET_GUI();
    return (nullptr != gui);
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

	while(ngui_isenabled())
	{
		uint8_t bits = 0;
		const Uint8 *kb = SDL_GetKeyboardState(NULL);
		
		for(size_t i = 0; i < (sizeof(bindings) / sizeof(bindings[0])); ++i)
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
	while(brain_on_frame())
	{
		uint8_t bits = brain_controller_bits();

		if(!brain_headless())
		{
		}

		njoypad_set(0, bits);
		nemu_frameadvance();
	}
	return 0;
}

int main(int argc, const char *argv[])
{
	SDL_SetMainReady();
	brain_init();

	if(argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <rom>" << std::endl;
		return 1;
	}

	if(!nemu_loadrom(argv[1]))
	{
		std::cout << "No rom :(" << std::endl;
		return 1;
	}

	if(!brain_headless())
	{
		if(!ngui_enable())
		{
			std::cout << "Unable to enable GUI" << std::endl;
			return 1;
		}
		ngui_setscale(2);
	}

	nemu_setframerate(0);

	if(brain_enabled())
	{
		brain_bind_cpu_mem(hqn_state.emu()->low_mem());
		return run_brain_mode();
	}

	return run_human_mode();
}



