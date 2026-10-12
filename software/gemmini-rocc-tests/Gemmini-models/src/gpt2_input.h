#ifndef GPT2_INPUT_H
#define GPT2_INPUT_H

#define IN_BATCH_INIT  1
#define SEQ_LEN_INIT   8
#define GEN_LEN        20
#define GEN_LEN_PAD    20

int input_token_ids[1][20] = {
    { 2061, 318, 262, 3139, 1748, 286, 2869, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

int input_attn_mask[1][20] = {
    { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

#endif // GPT2_INPUT_H
