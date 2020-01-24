#include "model.h"
#include "reward.h"
#include <curses.h>

void replay(Model &m)
{
    calculate_rewards(m);
    initscr();
    start_color();
    std::cout << "Color pairs: " << COLOR_PAIRS << std::endl;
    std::cout << "Colors: " << COLORS << std::endl;
    auto rgb = [](int r, int g, int b) {
        return 1 + r * 5 * 6 + g * 6 + b;
    };
    auto crgb = [&rgb](float r, float g, float b) {
        return rgb(r * 5, g * 6, b * 6);
    };
    float dcr = 1000.0 / 5.0;
    float dc = 1000.0 / 6.0;
    for (int r = 0; r < 5; ++r)
    {
        for (int g = 0; g < 6; ++g)
        {
            for (int b = 0; b < 6; ++b)
            {
                int ci = rgb(r, g, b);
                init_color(ci, r * dcr, g * dc, b * dc);
                init_pair(ci, rgb(5,5,5), ci);
            }
        }
    }
    cbreak();
    noecho();
    clear();
    char c = 0;
    int i = 0;
    do
    {
        clear();
        for (int sx = 0; sx < 32; ++sx)
        {
            for (int sy = 0; sy < 30; ++sy)
            {
                int screen_idx = sy * 32 * 3 + sx * 3;
                attron(COLOR_PAIR(crgb(
                    m.states[i][screen_idx + RAM_SIZE + 2] + 0.5,
                    m.states[i][screen_idx + RAM_SIZE + 1] + 0.5,
                    m.states[i][screen_idx + RAM_SIZE] + 0.5)));
                mvaddch(sy, 2 * sx, ' ');
                mvaddch(sy, 2 * sx + 1, ' ');
                // attroff(COLOR_PAIR(sx * 6));
                // m.states
            }
        }
        attron(COLOR_PAIR(1));
        mvprintw(0, 2 * 33, "frame: %d", i);
        mvprintw(1, 2 * 33, "reward: %f", m.rewards[i]);
        mvprintw(2, 2 * 33, "imm. reward: %f", m.immidiate_rewards[i]);

        std::string keys;
        keys.push_back(m.actions[i][0] ? 'U' : ' ');
        keys.push_back(m.actions[i][1] ? 'D' : ' ');
        keys.push_back(m.actions[i][2] ? 'L' : ' ');
        keys.push_back(m.actions[i][3] ? 'R' : ' ');
        keys.push_back(m.actions[i][4] ? 'A' : ' ');
        keys.push_back(m.actions[i][5] ? 'B' : ' ');
        keys.push_back(m.actions[i][6] ? 'S' : ' ');
        keys.push_back(m.actions[i][7] ? 'X' : ' ');

        mvprintw(3, 2 * 33, keys.c_str());

        refresh();
        c = getch();
        switch (c)
        {
        case 'L':
            i += 4;
        case 'l':
            i++;
            break;
        case 'H':
            i -= 4;
        case 'h':
            i--;
            break;
        case 'k':
            i += 50;
            break;
        case 'j':
            i -= 50;
            break;
        case ' ':
            i = 0;
            break;
        }
        for(int j = 1; j <= 9; ++j)
        {
            if(c == std::to_string(j)[0])
            {
                i = static_cast<int>((j / 9.0) * m.get_frames());
            }
        }
        if (i < 0)
            i = 0;
        if (i >= m.get_frames())
            i = m.get_frames() - 1;
    } while (c != 'q');
    endwin();
}