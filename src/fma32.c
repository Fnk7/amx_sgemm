/*
 *  an example shows how fma32 (amx op12) works
 */

#include <stdio.h>
#include <string.h>

#include "../dougallj/amx.h"

__attribute__((aligned(0x80))) float inputX[16];
__attribute__((aligned(0x80))) float inputY[16];
__attribute__((aligned(0x80))) float outputZ[64][16];

void ASM_FUNC()
{
    AMX_START();
    AMX_LDX((uint64_t)&inputX[0]);
    AMX_LDY((uint64_t)&inputY[0]);
    // maybe AMX_START init all registers with 0, 
    // but still use 27 to not add regiter Z
    AMX_FMA32(1ull << 27);
    for (uint64_t i = 0; i < 64; i++)
    {
        AMX_STZ((i << 56) | (uint64_t)&outputZ[i][0]);
    }
    AMX_STOP();
}

int main()
{
    for (int i = 0; i < 16; i++)
    {
        inputX[i] = (i % 5 + 1) / 2.0;
        inputY[i] = (i % 7 + 1) / 2.0;
    }
    for (int i = 0; i < 64; i++)
    {
        for (uint16_t j = 0; j < 16; j++)
        {
            outputZ[i][j] = 0;
        }
    }
    ASM_FUNC();
#if 0
    // print all result stored from amx register z
    for (int i = 0; i < 64; i++)
    {
        for (uint16_t j = 0; j < 16; j++)
        {
            printf("%.3f  ", outputZ[i][j]);
        }
        printf("\n");
    }
    printf("\n");
#endif
    for (int i = 0; i < 16; i++)
    {
        for (uint16_t j = 0; j < 16; j++)
        {
            printf("%.3f  ", outputZ[i * 4][j]);
        }
        printf("\n");
    }
}