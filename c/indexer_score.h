/* indexer_score.h -- the indexer's candidate scoring loop, header-only like c/quant.h.
 *
 * WHY: the indexer is ~3.3% of decode and its two heaviest loops run serially on a 16-thread
 * machine (c/deepseek_v4.c:3425-3431 and :3432-3442). Both write one independent output per
 * iteration, so parallelising over the outer index reorders nothing and is bit-exact.
 *
 * Extracted to a header so it can be unit-tested directly, without constructing a full
 * ColiDeepSeekV4Indexer (fp8 query weights, bf16 projection, compressor state). Same approach as
 * c/head_ilp.h.
 */
#ifndef COLI_V4_INDEXER_SCORE_H
#define COLI_V4_INDEXER_SCORE_H

#include <math.h>
#include <stddef.h>

/* score[candidate] = sum_head max(dot(query_head, key_candidate), 0) * head_weight[head] */
static inline void coli_v4_indexer_scores(float *scores, const float *queries,
                                          const float *compressed, const float *head_weights,
                                          int count, int heads, int dimension, int parallel) {
    if (parallel) {
        #pragma omp parallel for schedule(static)
        for (int candidate = 0; candidate < count; candidate++) {
            const float *key = compressed + (size_t)candidate * dimension;
            float score = 0.0f;
            for (int head = 0; head < heads; head++) {
                const float *query = queries + (size_t)head * dimension;
                float dot = 0.0f;
                for (int i = 0; i < dimension; i++) dot += query[i] * key[i];
                score += fmaxf(dot, 0.0f) * head_weights[head];
            }
            scores[candidate] = score;
        }
    } else {
        for (int candidate = 0; candidate < count; candidate++) {
            const float *key = compressed + (size_t)candidate * dimension;
            float score = 0.0f;
            for (int head = 0; head < heads; head++) {
                const float *query = queries + (size_t)head * dimension;
                float dot = 0.0f;
                for (int i = 0; i < dimension; i++) dot += query[i] * key[i];
                score += fmaxf(dot, 0.0f) * head_weights[head];
            }
            scores[candidate] = score;
        }
    }
}

#endif /* COLI_V4_INDEXER_SCORE_H */
