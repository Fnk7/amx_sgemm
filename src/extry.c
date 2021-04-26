/*
 *  test amx op 9, which is used for transpose
 */

#include <stdio.h>
#include <string.h>

#include "../dougallj/amx.h"

// use char to see diff
__attribute__((aligned(0x80))) char X[8][64];
__attribute__((aligned(0x80))) char Y[8][64];
__attribute__((aligned(0x80))) char Z[64][64];

void initZ()
{
    char *tZ = &Z[0][0];
    for (int i = 0; i < 64 * 64; i++)
    {
        int tmp = i % (26 + 10);
        tZ[i] = tmp < 26 ? 'a' + tmp : '0' + tmp - 26;
        if (i % 64 % 7 == 0 || i / 64 % 7 == 0)
            tZ[i] = ((i % 64 + i / 64) % 2 == 0) ? '*' : '#';
    }
}

void initXY()
{
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 64; j++)
        {
            X[i][j] = '+';
            Y[i][j] = '-';
        }
}

void printAMX(int reg)
{
    if (reg == 0)
    {
        printf("printZ\n");
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                printf("%c ", Z[i][j]);
            }
            printf("\n");
        }
    }
    else
    {
        char *tmp;
        if (reg == 1)
        {
            printf("printX\n");
            tmp = &X[0][0];
        }
        else
        {
            printf("printY\n");
            tmp = &Y[0][0];
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                printf("%c ", tmp[i * 64 + j]);
            }
            printf("\n");
        }
    }
}

void ASM_FUNC(uint64_t oprand)
{
    AMX_START();
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_LDX((i << 56) | (uint64_t)&X[i][0]);
    }
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_LDY((i << 56) | (uint64_t)&Y[i][0]);
    }
    for (uint64_t i = 0; i < 64; i++)
    {
        AMX_LDZ((i << 56) | (uint64_t)&Z[i][0]);
    }
    AMX_EXTRY(oprand);
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_STY((i << 56) | (uint64_t)&Y[i][0]);
    }
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_STX((i << 56) | (uint64_t)&X[i][0]);
    }
    AMX_STOP();
}

int offsets[] = {0, 20, 10, 11, 12, 13, 26, 27, 28, 29, 63};
int fileds = (sizeof(offsets) / sizeof(offsets[0]));
int setOprand(uint64_t *oprand, uint32_t idx, uint64_t val)
{
    if (idx >= fileds)
        return -1;
    uint64_t mask = idx == 0 ? 0x1FFull : idx == 1 ? 0x3Full
                                                   : 0x1ull;
    if (val > mask)
        return -1;
    int offset = offsets[idx];
    *oprand &= ~(mask << offset);
    *oprand |= (val << offset);
    return 0;
}

void printOprand(uint64_t oprand)
{
    printf("Oprand\t:\t0x%016llx\n", oprand);
    printf("Index\t:\t");
    for (int i = 0; i < fileds; i++)
    {
        printf("%2d\t", i);
    }
    printf("\n");
    printf("Offset\t:\t");
    printf(" y\t");
    printf(" z\t");
    for (int i = 2; i < fileds; i++)
    {
        printf("%2d\t", offsets[i]);
    }
    printf("\n");
    printf("Value\t:\t");
    printf("%d,%d\t", (int)(oprand & 0x1FF) / 64, (int)(oprand & 0x1FF) % 64);
    printf(" %d\t", (int)(oprand >> 20) & 0x3F);
    for (int i = 2; i < fileds; i++)
    {
        printf("%2d\t", (int)(oprand >> offsets[i]) & 0x1);
    }
    printf("\n");
}

int main()
{
    uint64_t oprand = 0x8000000004004000;
    char op;
    uint32_t idx;
    uint32_t val;
    op = 'r';
    do
    {
        if (op == 's')
        {
            printf("Enter index and val:\n");
            scanf("%d %d", &idx, &val);
            if (setOprand(&oprand, idx, val))
            {
                printf("%d, %d\n", idx, val);
            }
        }
        else if (op == 'r')
        {
            initZ();
            initXY();
            printAMX(0);
            ASM_FUNC(oprand);
            printAMX(1);
            printAMX(2);
        }
        else if (op == 'q')
        {
            return 0;
        }
        else
        {
            continue;
        }
        printOprand(oprand);
        printf("s set oprand, r run amx, q quit: \n");
    } while (scanf("%c", &op) == 1);
}