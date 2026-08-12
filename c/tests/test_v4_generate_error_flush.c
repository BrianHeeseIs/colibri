/* T3: the learning cache must be flushed on the GENERATE-ERROR path, not only on success.
 *
 * Why this test is shaped the way it is (three review rounds moved it here):
 *
 *  - Round 2: driving the failure with an oversized max_tokens does NOT work. That trips the
 *    CONTEXT_EXCEEDED precheck BEFORE coli_v4_session_generate() runs, so the epilogue is
 *    never reached and the test proves nothing. Hence the explicit fault-injection flag.
 *
 *  - Round 3: calling coli_v4_session_generate() directly does not traverse the v4_serve_one
 *    epilogue either. The test therefore calls coli_v4_test_flush_usage_epilogue(), a
 *    same-translation-unit forwarder to the SAME static v4_flush_usage_epilogue() body the
 *    serve path calls. Calling coli_v4_expert_store_flush_usage() directly is FORBIDDEN --
 *    it would bypass the epilogue and re-open exactly the gap this test exists to close.
 *
 *  - Round 3 (Oracle): asserting AFTER destroy is a FALSE POSITIVE, because destroy_hot()
 *    writes .coli_usage on teardown regardless of this feature. The existence assertion must
 *    therefore happen BEFORE any session or engine destroy. That ordering is the whole test.
 *
 * The injected failure fires AFTER prefill so lookup_hot() has already incremented
 * policy->usage -- the history written is non-vacuous rather than an empty file. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "../deepseek_v4.h"
#include "../deepseek_v4_internal.h"

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int on_token(void *user_data, int token, float logit, int position, int ordinal) {
    (void)user_data; (void)token; (void)logit; (void)position; (void)ordinal;
    return 0;   /* keep generating; the fault injection decides when we stop */
}

int main(int argc, char **argv) {
    const char *model_dir = argc > 1 ? argv[1] : "deepseek_v4_tiny";
    char history[1024];
    snprintf(history, sizeof(history), "%s/.coli_usage", model_dir);

    char error[512] = {0};

    /* 1. fresh engine + fresh session (no prefix reuse, so prefill really does look up
     *    experts and therefore really does increment policy->usage). */
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions open_options = {.target_model_dir = model_dir};
    if (coli_v4_engine_open(&engine, &open_options, error, sizeof(error))) {
        fprintf(stderr, "error-path flush: cannot open engine on %s: %s\n", model_dir, error);
        return 1;
    }

    ColiV4Session *session = NULL;
    ColiV4SessionCreateOptions session_options = {0};
    if (coli_v4_session_create(&session, engine, &session_options, error, sizeof(error))) {
        fprintf(stderr, "error-path flush: cannot create session: %s\n", error);
        coli_v4_engine_destroy(engine);
        return 1;
    }

    /* 2. start from a known-absent history file. */
    unlink(history);
    assert(!file_exists(history) && "history must be absent before the run");

    /* 3. force generate to fail after prefill. */
    coli_v4_test_fail_generate_after_prefill = 1;

    /* 4. generation must report failure. */
    ColiV4SessionGenerateStats stats = {0};
    ColiV4SessionGenerateOptions generate_options = {
        .max_new_tokens = 4, .stop_at_sentence = 0, .no_dspark = 1,
    };
    const char *prompt = "hi";
    int result = coli_v4_session_generate(session, prompt, strlen(prompt),
                                          &generate_options, on_token, NULL,
                                          &stats, error, sizeof(error));
    coli_v4_test_fail_generate_after_prefill = 0;
    assert(result != 0 && "fault injection must make generate fail");

    /* 5. run the PRODUCTION epilogue body (via the test-only forwarder). */
    coli_v4_test_flush_usage_epilogue(engine);

    /* 6. THE assertion -- strictly before any destroy, or destroy_hot() would write the
     *    file for us and this test would pass even with the feature removed. */
    int flushed = file_exists(history);

    /* 7. only now tear down. */
    coli_v4_session_destroy(session);
    coli_v4_engine_destroy(engine);

    /* Leave the fixture as we found it. This must happen AFTER the destroys, because
     * destroy_hot() writes .coli_usage again on teardown -- unlinking earlier would just
     * see the file recreated. The pass/fail decision was already captured in `flushed`
     * above, so removing the file here cannot weaken the assertion.
     * (.coli_usage is not gitignored, so without this every `make test-c` would leave an
     * untracked artifact in c/deepseek_v4_tiny/.) */
    unlink(history);

    if (!flushed) {
        fprintf(stderr, "error-path flush: FAILED -- %s absent after the epilogue\n", history);
        return 1;
    }
    printf("error-path flush: ok\n");
    return 0;
}
