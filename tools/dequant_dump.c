// dequant_dump.c -- dump ggml's OWN dequantization of raw Q4_K / Q6_K / Q8_0 blocks.
//   usage: dequant_dump <q4_k|q5_k|q6_k|q8_0> <raw_blocks.bin> <out_f32.bin>
//   Reads N whole blocks (144B q4_k / 176B q5_k / 210B q6_k per 256 weights,
//   34B q8_0 per 32 weights), calls the reference dequantize_row_* from ggml
//   (the exact code llama.cpp runs), writes fp32.
//   Build: tools/build_dequant_dump.sh (links against llama.cpp's libggml-base).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ggml reference dequant prototypes (ggml/src/ggml-quants.h)
void dequantize_row_q4_K(const void * x, float * y, long k);
void dequantize_row_q5_K(const void * x, float * y, long k);
void dequantize_row_q6_K(const void * x, float * y, long k);
void dequantize_row_q8_0(const void * x, float * y, long k);

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <q4_k|q5_k|q6_k|q8_0> <in> <out>\n", argv[0]); return 2; }
    size_t bs; long wpb;                 // bytes per block, weights per block
    if      (strcmp(argv[1], "q4_k") == 0) { bs = 144; wpb = 256; }
    else if (strcmp(argv[1], "q5_k") == 0) { bs = 176; wpb = 256; }
    else if (strcmp(argv[1], "q6_k") == 0) { bs = 210; wpb = 256; }
    else if (strcmp(argv[1], "q8_0") == 0) { bs = 34;  wpb = 32;  }
    else { fprintf(stderr, "unknown type %s\n", argv[1]); return 2; }
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) { perror("in"); return 1; }
    fseek(fi, 0, SEEK_END); long sz = ftell(fi); fseek(fi, 0, SEEK_SET);
    if (sz % bs) { fprintf(stderr, "size %ld not multiple of %zu\n", sz, bs); return 1; }
    long nb = sz / bs, k = nb * wpb;
    void *raw = malloc(sz); float *out = malloc(k * sizeof(float));
    if (fread(raw, 1, sz, fi) != (size_t)sz) { perror("read"); return 1; }
    fclose(fi);
    if      (strcmp(argv[1], "q5_k") == 0) dequantize_row_q5_K(raw, out, k);
    else if (strcmp(argv[1], "q6_k") == 0) dequantize_row_q6_K(raw, out, k);
    else if (strcmp(argv[1], "q8_0") == 0) dequantize_row_q8_0(raw, out, k);
    else                                   dequantize_row_q4_K(raw, out, k);
    FILE *fo = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), k, fo);
    fclose(fo);
    fprintf(stderr, "dequantized %ld blocks (%ld weights) as %s\n", nb, k, argv[1]);
    return 0;
}
