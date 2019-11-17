#include "hqn.h"
#include "hqn_gui_controller.h"

#include "gif.h"

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

static bool is_human = false;

static const char *gif = nullptr;
static GifWriter g;

static int frame = 0;

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

static void nemu_hard_reset()
{
	hqn_state.reset(true);
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

static int32_t frame_pixels[256 * 240];

static int run_brain()
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

	while(brain_on_frame())
	{
		uint8_t bits = brain_controller_bits();

		if(is_human)
		{
			if(!ngui_isenabled())
			{
				break;
			}

			const Uint8 *kb = SDL_GetKeyboardState(NULL);
		
			bits = 0;

			for(size_t i = 0; i < (sizeof(bindings) / sizeof(bindings[0])); ++i)
			{
				const auto &b = bindings[i];
				if(kb[b.sdl_scancode])
				{
					bits |= b.bit;
				}
			}
		}

		if(!brain_headless())
		{
		}

		njoypad_set(0, bits);
		nemu_frameadvance();
		frame++;
		if(gif && !(frame&0xf))
		{
			hqn_state.blit(frame_pixels, HQNState::NES_VIDEO_PALETTE, 0, 0, 0, 0);
			for(int i = 0; i < 256*240; ++i) {
				auto r = frame_pixels[i] >> 16 & 0xff;
				auto g = frame_pixels[i] >> 8 & 0xff;
				auto b = frame_pixels[i] & 0xff;
				frame_pixels[i] = 0xff000000 | (b << 16) | (g << 8) | (r);
			}
			GifWriteFrame(&g, reinterpret_cast<uint8_t *>(frame_pixels), 256, 240, 1);
		}
	}
	return 0;
}

static int get_frame_rate()
{
	if(is_human)
	{
		return 60;
	}

	const char *fr = getenv("FPS");
	if(fr)
	{
		return atoi(fr);
	}
	return 0;
}

int main(int argc, const char *argv[])
{
	SDL_SetMainReady();
	brain_init();

	if(!brain_enabled())
	{
		std::cout << "Brain not enabled, use BE=<script.lua>" << std::endl;
		return 1;
	}

	if(argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <rom>" << std::endl;
		return 1;
	}

	gif = getenv("GIF");

	if(gif)
	{
		GifBegin(&g, gif, 256, 240, 1);
	}

	const char * const human = getenv("HUMAN");
	is_human = (human && '1' == *human);

	if(is_human && brain_headless())
	{
		std::cout << "Cant run human mode in headless. Remove HL=1" << std::endl;
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

	nemu_setframerate(get_frame_rate());

	uint32_t rollout = 0;

	for(;;)
	{
		brain_bind_cpu_mem(hqn_state.emu()->low_mem());

		int rc = run_brain();

		if(0 != rc)
		{
			std::cout << "Brain failed to run with: " << rc << std::endl;
			return rc;
		}

		if(++rollout == brain_num_rollouts())
		{
			std::cout << "Ran " << rollout << " rollouts successfully! :)" << std::endl;
			break;
		}

		nemu_hard_reset();
	}

	if(gif)
	{
		GifEnd(&g);
	}

	return 0;
}



