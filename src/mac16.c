/*
 *  an example shows how mac16 (amx op14) works
 *  the op14 is described by dougallj with great details
 */

#include <stdio.h>
#include <string.h>

#include "../dougallj/amx.h"


__attribute__((aligned(0x80))) int16_t inputX[32];
__attribute__((aligned(0x80))) int16_t inputY[32];
__attribute__((aligned(0x80))) int16_t outputZ[32][32];

void ASM_FUNC()
{
    AMX_START();
    AMX_LDX((uint64_t)&inputX[0]);
    AMX_LDY((uint64_t)&inputY[0]);
    // maybe AMX_START (op17) init all registers with 0.
    // I guess the op17 give a init signal to amx
    AMX_MAC16(1l << 27);
    // repeat
    AMX_MAC16(0);
    for (uint64_t i = 0; i < 64; i += 2)
    {
        AMX_STZ((i << 56) | (uint64_t)&outputZ[i >> 1][0]);
    }
    AMX_STOP();
}

int main()
{
    for (int16_t i = 0; i < 32; i++)
    {
        inputX[i] = i % 3 + 1;
        inputY[i] = i % 4 + 1;
        for (int16_t j = 0; j < 32; j++)
        {
            outputZ[i][j] = 0;
        }
    }
    ASM_FUNC();
    for (int16_t i = 0; i < 32; i++)
    {
        for (int16_t j = 0; j < 32; j++)
        {
            printf("%2d ", outputZ[i][j]);
        }
        printf("\n");
    }
}