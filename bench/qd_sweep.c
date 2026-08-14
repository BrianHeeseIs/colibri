/*
 * DeepSeek V4 Flash SSD queue-depth sweep.
 *
 * Build on macOS with project-equivalent clang flags (no -ffast-math):
 *   clang -D_DARWIN_C_SOURCE -D_FILE_OFFSET_BITS=64 -O3 -pthread -Wall -Wextra \
 *     -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function \
 *     bench/qd_sweep.c -o bench/qd_sweep
 *
 * Normal run (exclusive disk):
 *   ./bench/qd_sweep models/deepseek-v4-flash [artifacts/layer_contig.json]
 *
 * QD_SWEEP_SMOKE=1 runs one discarded + one timed QD1 round over two records.
 * It validates discovery and direct reads, but deliberately emits no GATE_A.
 */
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#define _FILE_OFFSET_BITS 64

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    ALIGNMENT = 4096,
    BUFFER_ALIGNMENT = 16384,
    DEFAULT_RECORDS = 256,
    SMOKE_RECORDS = 2,
    TIMED_ROUNDS = 3,
    MAX_HEADER_BYTES = 512 * 1024 * 1024,
};

static const uint64_t EXPECTED_RECORD_BYTES = UINT64_C(13369344);
static const uint64_t SELECT_SEED = UINT64_C(0x6d0f27a38db4c251);

typedef struct {
    char *path;
    int fd;
    int tail_fd;
    uint64_t size;
    bool needed;
} Shard;

typedef struct {
    Shard *items;
    size_t count;
    size_t capacity;
} ShardList;

typedef struct {
    int layer;
    int expert;
    size_t shard;
    uint64_t weight_offset;
    uint64_t weight_len;
    uint64_t scale_offset;
    uint64_t scale_len;
} Record;

typedef struct {
    Record *items;
    size_t count;
    size_t capacity;
} RecordList;

typedef struct {
    int layer;
    int expert;
    int matrix;
    bool scale;
    uint64_t start;
    uint64_t end;
} Part;

typedef struct {
    Part *items;
    size_t count;
    size_t capacity;
} PartList;

typedef struct Pool Pool;

typedef struct {
    Pool *pool;
    unsigned char *buffer;
} Worker;

struct Pool {
    Worker *workers;
    pthread_t *threads;
    int worker_count;
    pthread_mutex_t mutex;
    pthread_cond_t start;
    pthread_cond_t done;
    unsigned generation;
    int active;
    bool stop;
    _Atomic size_t next;
    _Atomic bool failed;
    const ShardList *shards;
    const Record *const *work;
    size_t work_count;
    double *latencies_ms;
    char error[512];
    size_t buffer_bytes;
};

typedef struct {
    double gbps;
    double p50_ms;
    double p99_ms;
} RoundStats;

static void die(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fputs("qd_sweep: ", stderr);
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t bytes) {
    void *value = malloc(bytes);
    if (!value) die("out of memory allocating %zu bytes", bytes);
    return value;
}

static void *xrealloc(void *pointer, size_t bytes) {
    void *value = realloc(pointer, bytes);
    if (!value) die("out of memory reallocating %zu bytes", bytes);
    return value;
}

static char *xstrdup(const char *text) {
    size_t bytes = strlen(text) + 1;
    char *copy = xmalloc(bytes);
    memcpy(copy, text, bytes);
    return copy;
}

static char *xstrndup(const char *text, size_t length) {
    char *copy = xmalloc(length + 1);
    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

static uint64_t read_le64(const unsigned char bytes[8]) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) value |= (uint64_t)bytes[i] << (8 * i);
    return value;
}

static bool has_suffix(const char *text, const char *suffix) {
    size_t text_length = strlen(text);
    size_t suffix_length = strlen(suffix);
    return text_length >= suffix_length &&
        memcmp(text + text_length - suffix_length, suffix, suffix_length) == 0;
}

static void grow_array(void **items, size_t *capacity, size_t item_size) {
    size_t next = *capacity ? *capacity * 2 : 64;
    if (next > SIZE_MAX / item_size) die("array size overflow");
    *items = xrealloc(*items, next * item_size);
    *capacity = next;
}

static int compare_shards(const void *left, const void *right) {
    const Shard *a = left;
    const Shard *b = right;
    return strcmp(a->path, b->path);
}

static int compare_parts(const void *left, const void *right) {
    const Part *a = left;
    const Part *b = right;
    if (a->layer != b->layer) return (a->layer > b->layer) - (a->layer < b->layer);
    if (a->expert != b->expert) return (a->expert > b->expert) - (a->expert < b->expert);
    if (a->matrix != b->matrix) return (a->matrix > b->matrix) - (a->matrix < b->matrix);
    return (int)a->scale - (int)b->scale;
}

static int compare_records(const void *left, const void *right) {
    const Record *a = left;
    const Record *b = right;
    if (a->layer != b->layer) return (a->layer > b->layer) - (a->layer < b->layer);
    return (a->expert > b->expert) - (a->expert < b->expert);
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static void pread_full(int fd, void *destination, size_t length, uint64_t offset,
                       const char *what) {
    unsigned char *output = destination;
    size_t done = 0;
    while (done < length) {
        ssize_t count = pread(fd, output + done, length - done,
                              (off_t)(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            if (count < 0) die("%s at offset %llu: %s", what,
                               (unsigned long long)(offset + done), strerror(errno));
            die("%s short read at offset %llu (%zu of %zu bytes)", what,
                (unsigned long long)(offset + done), done, length);
        }
        done += (size_t)count;
    }
}

static uint64_t file_size(const char *path) {
    struct stat status;
    if (stat(path, &status) != 0) die("stat %s: %s", path, strerror(errno));
    if (status.st_size < 0) die("negative file size for %s", path);
    return (uint64_t)status.st_size;
}

static char *join_path(const char *directory, const char *name) {
    if (name[0] == '/') return xstrdup(name);
    size_t directory_length = strlen(directory);
    size_t name_length = strlen(name);
    if (directory_length > SIZE_MAX - name_length - 2) die("path length overflow");
    char *path = xmalloc(directory_length + name_length + 2);
    snprintf(path, directory_length + name_length + 2, "%s/%s", directory, name);
    return path;
}

static size_t shard_add(ShardList *shards, const char *path) {
    for (size_t i = 0; i < shards->count; i++)
        if (strcmp(shards->items[i].path, path) == 0) return i;
    if (shards->count == shards->capacity)
        grow_array((void **)&shards->items, &shards->capacity, sizeof(*shards->items));
    Shard *shard = &shards->items[shards->count];
    *shard = (Shard){.path = xstrdup(path), .fd = -1, .tail_fd = -1,
                     .size = file_size(path), .needed = false};
    return shards->count++;
}

static void record_validate(const Record *record, const ShardList *shards) {
    if (record->shard >= shards->count) die("record has unknown shard");
    const Shard *shard = &shards->items[record->shard];
    if (!record->weight_len || !record->scale_len ||
        record->weight_offset > shard->size ||
        record->weight_len > shard->size - record->weight_offset ||
        record->scale_offset > shard->size ||
        record->scale_len > shard->size - record->scale_offset)
        die("out-of-bounds expert range layer=%d expert=%d in %s",
            record->layer, record->expert, shard->path);
    if (record->weight_len > UINT64_MAX - record->scale_len)
        die("record byte count overflow layer=%d expert=%d", record->layer, record->expert);
    uint64_t total = record->weight_len + record->scale_len;
    uint64_t lower = EXPECTED_RECORD_BYTES * 99 / 100;
    uint64_t upper = EXPECTED_RECORD_BYTES * 101 / 100;
    if (total < lower || total > upper)
        die("unexpected record size layer=%d expert=%d: %llu bytes, expected about %llu",
            record->layer, record->expert, (unsigned long long)total,
            (unsigned long long)EXPECTED_RECORD_BYTES);
}

static void record_add(RecordList *records, const Record *record,
                       const ShardList *shards) {
    record_validate(record, shards);
    if (records->count == records->capacity)
        grow_array((void **)&records->items, &records->capacity, sizeof(*records->items));
    records->items[records->count++] = *record;
}

static const char *skip_ws(const char *cursor, const char *end) {
    while (cursor < end && (*cursor == ' ' || *cursor == '\t' ||
                            *cursor == '\r' || *cursor == '\n')) cursor++;
    return cursor;
}

static const char *string_end(const char *start, const char *end) {
    if (start >= end || *start != '"') return NULL;
    for (const char *cursor = start + 1; cursor < end; cursor++) {
        if (*cursor == '\\') {
            if (++cursor >= end) return NULL;
        } else if (*cursor == '"') {
            return cursor;
        }
    }
    return NULL;
}

static const char *object_end(const char *start, const char *end) {
    if (start >= end || *start != '{') return NULL;
    int depth = 0;
    for (const char *cursor = start; cursor < end; cursor++) {
        if (*cursor == '"') {
            cursor = string_end(cursor, end);
            if (!cursor) return NULL;
        } else if (*cursor == '{') {
            depth++;
        } else if (*cursor == '}' && --depth == 0) {
            return cursor;
        }
    }
    return NULL;
}

static bool key_matches(const char *start, const char *finish, const char *key) {
    size_t length = strlen(key);
    return (size_t)(finish - start - 1) == length &&
        memcmp(start + 1, key, length) == 0;
}

static const char *field_value(const char *object, const char *end, const char *key) {
    for (const char *cursor = object + 1; cursor < end; cursor++) {
        if (*cursor != '"') continue;
        const char *finish = string_end(cursor, end);
        if (!finish) return NULL;
        const char *after = skip_ws(finish + 1, end);
        if (after < end && *after == ':' && key_matches(cursor, finish, key))
            return skip_ws(after + 1, end);
        cursor = finish;
    }
    return NULL;
}

static bool parse_u64(const char *cursor, const char *end, uint64_t *value) {
    cursor = skip_ws(cursor, end);
    if (cursor >= end || *cursor < '0' || *cursor > '9') return false;
    uint64_t parsed = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        unsigned digit = (unsigned)(*cursor - '0');
        if (parsed > (UINT64_MAX - digit) / 10) return false;
        parsed = parsed * 10 + digit;
        cursor++;
    }
    *value = parsed;
    return true;
}

static bool parse_record_key(const char *text, size_t length, int *layer, int *expert) {
    if (length >= 64) return false;
    char copy[64];
    memcpy(copy, text, length);
    copy[length] = '\0';
    int used = 0;
    return sscanf(copy, "%d:%d%n", layer, expert, &used) == 2 &&
        (size_t)used == length && *layer >= 0 && *expert >= 0;
}

static bool parse_offsets(const char *object, const char *end,
                          uint64_t *start, uint64_t *finish) {
    const char *cursor = field_value(object, end, "data_offsets");
    if (!cursor || cursor >= end || *cursor != '[') return false;
    if (!parse_u64(cursor + 1, end, start)) return false;
    while (cursor < end && *cursor != ',') cursor++;
    if (cursor == end || !parse_u64(cursor + 1, end, finish)) return false;
    while (cursor < end && *cursor != ']') cursor++;
    return cursor < end && *finish >= *start;
}

static bool parse_expert_name(const char *name, size_t length, int *layer,
                              int *expert, int *matrix, bool *scale) {
    if (length >= 192) return false;
    char copy[192];
    memcpy(copy, name, length);
    copy[length] = '\0';
    char kind[16];
    int used = 0;
    if (sscanf(copy, "layers.%d.ffn.experts.%d.w%d.%15s%n",
               layer, expert, matrix, kind, &used) != 4 ||
        (size_t)used != length || *layer < 0 || *expert < 0 ||
        *matrix < 1 || *matrix > 3)
        return false;
    if (strcmp(kind, "scale") == 0) *scale = true;
    else if (strcmp(kind, "weight") == 0) *scale = false;
    else return false;
    return true;
}

static void part_add(PartList *parts, const Part *part) {
    if (parts->count == parts->capacity)
        grow_array((void **)&parts->items, &parts->capacity, sizeof(*parts->items));
    parts->items[parts->count++] = *part;
}

static void scan_header_parts(const Shard *shard, PartList *parts) {
    int fd = open(shard->path, O_RDONLY);
    if (fd < 0) die("open %s: %s", shard->path, strerror(errno));
    unsigned char prefix[8];
    pread_full(fd, prefix, sizeof(prefix), 0, "read safetensors header length");
    uint64_t header_length = read_le64(prefix);
    if (shard->size < 8 || !header_length || header_length > MAX_HEADER_BYTES ||
        header_length > shard->size - 8)
        die("invalid safetensors header length in %s", shard->path);
    char *header = xmalloc((size_t)header_length + 1);
    pread_full(fd, header, (size_t)header_length, 8, "read safetensors header");
    close(fd);
    header[header_length] = '\0';
    const char *end = header + header_length;
    for (const char *cursor = header; cursor < end; cursor++) {
        if (*cursor != '"') continue;
        const char *finish = string_end(cursor, end);
        if (!finish) die("malformed JSON string in %s", shard->path);
        const char *value = skip_ws(finish + 1, end);
        if (value >= end || *value != ':') {
            cursor = finish;
            continue;
        }
        value = skip_ws(value + 1, end);
        if (value >= end || *value != '{') {
            cursor = finish;
            continue;
        }
        int layer, expert, matrix;
        bool scale;
        if (parse_expert_name(cursor + 1, (size_t)(finish - cursor - 1),
                              &layer, &expert, &matrix, &scale)) {
            const char *object_finish = object_end(value, end);
            uint64_t relative_start = 0, relative_end = 0;
            if (!object_finish || !parse_offsets(value, object_finish,
                                                &relative_start, &relative_end))
                die("malformed data_offsets for layer=%d expert=%d in %s",
                    layer, expert, shard->path);
            if (relative_end > UINT64_MAX - 8 - header_length ||
                8 + header_length + relative_end > shard->size)
                die("out-of-bounds tensor layer=%d expert=%d in %s",
                    layer, expert, shard->path);
            part_add(parts, &(Part){
                .layer = layer, .expert = expert, .matrix = matrix, .scale = scale,
                .start = 8 + header_length + relative_start,
                .end = 8 + header_length + relative_end,
            });
        }
        cursor = finish;
    }
    free(header);
}

static void range_from_parts(const Part *group, bool scale,
                             uint64_t *offset, uint64_t *length,
                             int layer, int expert) {
    const Part *matrices[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 6; i++) {
        if (group[i].scale != scale) continue;
        int matrix = group[i].matrix - 1;
        if (matrix < 0 || matrix >= 3 || matrices[matrix])
            die("duplicate or invalid matrix layer=%d expert=%d", layer, expert);
        matrices[matrix] = &group[i];
    }
    if (!matrices[0] || !matrices[1] || !matrices[2])
        die("missing matrix layer=%d expert=%d", layer, expert);
    const Part *ordered[3] = {matrices[0], matrices[1], matrices[2]};
    for (int i = 0; i < 3; i++) for (int j = i + 1; j < 3; j++)
        if (ordered[i]->start > ordered[j]->start) {
            const Part *swap = ordered[i]; ordered[i] = ordered[j]; ordered[j] = swap;
        }
    if (ordered[0]->end != ordered[1]->start || ordered[1]->end != ordered[2]->start)
        die("non-contiguous %s range layer=%d expert=%d",
            scale ? "scale" : "weight", layer, expert);
    *offset = ordered[0]->start;
    *length = ordered[2]->end - ordered[0]->start;
}

static void extract_records(ShardList *shards, RecordList *records) {
    for (size_t shard_index = 0; shard_index < shards->count; shard_index++) {
        PartList parts = {0};
        scan_header_parts(&shards->items[shard_index], &parts);
        qsort(parts.items, parts.count, sizeof(*parts.items), compare_parts);
        for (size_t start = 0; start < parts.count;) {
            size_t finish = start + 1;
            while (finish < parts.count &&
                   parts.items[finish].layer == parts.items[start].layer &&
                   parts.items[finish].expert == parts.items[start].expert) finish++;
            if (finish - start != 6)
                die("expected six tensor parts layer=%d expert=%d in %s",
                    parts.items[start].layer, parts.items[start].expert,
                    shards->items[shard_index].path);
            Record record = {.layer = parts.items[start].layer,
                             .expert = parts.items[start].expert,
                             .shard = shard_index};
            range_from_parts(&parts.items[start], true, &record.scale_offset,
                             &record.scale_len, record.layer, record.expert);
            range_from_parts(&parts.items[start], false, &record.weight_offset,
                             &record.weight_len, record.layer, record.expert);
            record_add(records, &record, shards);
            start = finish;
        }
        free(parts.items);
    }
    if (!records->count) die("no DeepSeek V4 routed-expert records found");
}

static bool parse_int_field(const char *object, const char *end, const char *key,
                            int *value) {
    uint64_t parsed;
    const char *cursor = field_value(object, end, key);
    if (!cursor || !parse_u64(cursor, end, &parsed) || parsed > INT_MAX) return false;
    *value = (int)parsed;
    return true;
}

static bool parse_u64_field(const char *object, const char *end, const char *key,
                            uint64_t *value) {
    const char *cursor = field_value(object, end, key);
    return cursor && parse_u64(cursor, end, value);
}

static bool parse_string_field(const char *object, const char *end, const char *key,
                               char **value) {
    const char *cursor = field_value(object, end, key);
    if (!cursor || cursor >= end || *cursor != '"') return false;
    const char *finish = string_end(cursor, end);
    if (!finish || memchr(cursor + 1, '\\', (size_t)(finish - cursor - 1))) return false;
    *value = xstrndup(cursor + 1, (size_t)(finish - cursor - 1));
    return true;
}

static bool parse_artifact_record(const char *object, const char *end,
                                  const char *model_dir, ShardList *shards,
                                  RecordList *records, int key_layer,
                                  int key_expert) {
    Record record = {0};
    char *shard_name = NULL;
    bool keyed_record = key_layer >= 0 && key_expert >= 0;
    if (!keyed_record && !field_value(object, end, "layer")) return false;
    if (keyed_record) {
        record.layer = key_layer;
        record.expert = key_expert;
    } else if (!parse_int_field(object, end, "layer", &record.layer) ||
               !parse_int_field(object, end, "expert", &record.expert)) {
        die("malformed layer_contig record");
    }
    if ((!parse_string_field(object, end, "shard", &shard_name) &&
         !parse_string_field(object, end, "shard_file", &shard_name) &&
         !parse_string_field(object, end, "shard_filename", &shard_name) &&
         !parse_string_field(object, end, "file", &shard_name)) ||
        !parse_u64_field(object, end, "weight_offset", &record.weight_offset) ||
        !(parse_u64_field(object, end, "weight_len", &record.weight_len) ||
          parse_u64_field(object, end, "weight_bytes", &record.weight_len)) ||
        !parse_u64_field(object, end, "scale_offset", &record.scale_offset) ||
        !(parse_u64_field(object, end, "scale_len", &record.scale_len) ||
          parse_u64_field(object, end, "scale_bytes", &record.scale_len)))
        die("malformed layer_contig record");
    char *path = join_path(model_dir, shard_name);
    record.shard = shard_add(shards, path);
    free(path);
    free(shard_name);
    record_add(records, &record, shards);
    return true;
}

static void load_artifact(const char *path, const char *model_dir,
                          ShardList *shards, RecordList *records) {
    uint64_t size = file_size(path);
    if (size > 64 * 1024 * 1024) die("artifact %s is unexpectedly large", path);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("open %s: %s", path, strerror(errno));
    char *text = xmalloc((size_t)size + 1);
    pread_full(fd, text, (size_t)size, 0, "read layer_contig artifact");
    close(fd);
    text[size] = '\0';
    const char *end = text + size;
    for (const char *cursor = text; cursor < end; cursor++) {
        if (*cursor != '"') continue;
        const char *key_finish = string_end(cursor, end);
        if (!key_finish) die("malformed JSON string in %s", path);
        const char *value = skip_ws(key_finish + 1, end);
        if (value >= end || *value != ':') {
            cursor = key_finish;
            continue;
        }
        value = skip_ws(value + 1, end);
        if (value >= end || *value != '{') {
            cursor = key_finish;
            continue;
        }
        const char *object_finish = object_end(value, end);
        if (!object_finish) die("malformed JSON object in %s", path);
        int layer = -1, expert = -1;
        (void)parse_record_key(cursor + 1, (size_t)(key_finish - cursor - 1),
                               &layer, &expert);
        (void)parse_artifact_record(value, object_finish, model_dir, shards, records,
                                    layer, expert);
        cursor = key_finish;
    }
    free(text);
    if (!records->count) die("artifact %s contains no usable records", path);
}

static void scan_shard_directory(const char *model_dir, ShardList *shards) {
    DIR *directory = opendir(model_dir);
    if (!directory) die("opendir %s: %s", model_dir, strerror(errno));
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (!has_suffix(entry->d_name, ".safetensors")) continue;
        char *path = join_path(model_dir, entry->d_name);
        (void)shard_add(shards, path);
        free(path);
    }
    closedir(directory);
    if (!shards->count) die("no .safetensors shards in %s", model_dir);
    qsort(shards->items, shards->count, sizeof(*shards->items), compare_shards);
}

static uint64_t rng_next(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static void shuffle_records(const Record **records, size_t count, uint64_t seed) {
    for (size_t i = count; i > 1; i--) {
        size_t other = (size_t)(rng_next(&seed) % i);
        const Record *swap = records[i - 1];
        records[i - 1] = records[other];
        records[other] = swap;
    }
}

static void require_unique_records(RecordList *records) {
    qsort(records->items, records->count, sizeof(*records->items), compare_records);
    for (size_t i = 1; i < records->count; i++)
        if (records->items[i - 1].layer == records->items[i].layer &&
            records->items[i - 1].expert == records->items[i].expert)
            die("duplicate record layer=%d expert=%d", records->items[i].layer,
                records->items[i].expert);
}

static const Record **select_work(const RecordList *records, size_t count,
                                  bool require_breadth) {
    if (records->count < count)
        die("need %zu distinct records, found %zu", count, records->count);
    const Record **candidates = xmalloc(records->count * sizeof(*candidates));
    for (size_t i = 0; i < records->count; i++) candidates[i] = &records->items[i];
    shuffle_records(candidates, records->count, SELECT_SEED);
    const Record **selected = xmalloc(count * sizeof(*selected));
    memcpy(selected, candidates, count * sizeof(*selected));
    free(candidates);
    if (!require_breadth) return selected;
    bool layers[256] = {false};
    size_t layer_count = 0;
    size_t first_shard = selected[0]->shard;
    bool multiple_shards = false;
    for (size_t i = 0; i < count; i++) {
        if (selected[i]->layer < 0 || selected[i]->layer >= 256)
            die("layer number outside benchmark bounds");
        if (!layers[selected[i]->layer]) {
            layers[selected[i]->layer] = true;
            layer_count++;
        }
        if (selected[i]->shard != first_shard) multiple_shards = true;
    }
    if (layer_count < 8 || !multiple_shards)
        die("selected work lacks required breadth: %zu layers, %s shards",
            layer_count, multiple_shards ? "multiple" : "one");
    return selected;
}

static size_t round_up(size_t value, size_t alignment) {
    if (value > SIZE_MAX - (alignment - 1)) die("aligned size overflow");
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t max_buffer_bytes(const Record *const *work, size_t count) {
    size_t maximum = 0;
    for (size_t i = 0; i < count; i++) {
        const Record *record = work[i];
        uint64_t scale_needed = (record->scale_offset & (ALIGNMENT - 1)) + record->scale_len;
        uint64_t weight_needed = (record->weight_offset & (ALIGNMENT - 1)) + record->weight_len;
        uint64_t record_needed = scale_needed;
        if (record->scale_offset + record->scale_len == record->weight_offset) {
            if (record->scale_len > UINT64_MAX - record->weight_len)
                die("buffer size overflow");
            record_needed = (record->scale_offset & (ALIGNMENT - 1)) +
                record->scale_len + record->weight_len;
        }
        if (scale_needed > SIZE_MAX || weight_needed > SIZE_MAX || record_needed > SIZE_MAX)
            die("buffer size overflow");
        size_t scale_bytes = round_up((size_t)scale_needed, ALIGNMENT);
        size_t weight_bytes = round_up((size_t)weight_needed, ALIGNMENT);
        size_t record_bytes = round_up((size_t)record_needed, ALIGNMENT);
        if (scale_bytes > maximum) maximum = scale_bytes;
        if (weight_bytes > maximum) maximum = weight_bytes;
        if (record_bytes > maximum) maximum = record_bytes;
    }
    return maximum;
}

static void open_needed_shards(ShardList *shards, const Record *const *work,
                               size_t count) {
    for (size_t i = 0; i < count; i++) shards->items[work[i]->shard].needed = true;
    for (size_t i = 0; i < shards->count; i++) {
        Shard *shard = &shards->items[i];
        if (!shard->needed) continue;
#ifdef __APPLE__
        shard->fd = open(shard->path, O_RDONLY);
        shard->tail_fd = shard->fd;
        if (shard->fd < 0) die("open %s: %s", shard->path, strerror(errno));
        if (fcntl(shard->fd, F_NOCACHE, 1) < 0)
            die("fcntl(F_NOCACHE) %s: %s", shard->path, strerror(errno));
#elif defined(O_DIRECT)
        shard->fd = open(shard->path, O_RDONLY | O_DIRECT);
        shard->tail_fd = open(shard->path, O_RDONLY);
        if (shard->fd < 0 || shard->tail_fd < 0)
            die("open direct %s: %s", shard->path, strerror(errno));
#else
        die("this benchmark requires macOS F_NOCACHE or O_DIRECT");
#endif
    }
}

static void close_shards(ShardList *shards) {
    for (size_t i = 0; i < shards->count; i++) {
        if (shards->items[i].tail_fd >= 0 && shards->items[i].tail_fd != shards->items[i].fd)
            close(shards->items[i].tail_fd);
        if (shards->items[i].fd >= 0) close(shards->items[i].fd);
        free(shards->items[i].path);
    }
    free(shards->items);
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        die("clock_gettime: %s", strerror(errno));
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static bool first_eight_zero(const unsigned char *buffer, size_t length) {
    if (length < 8) return true;
    for (size_t i = 0; i < 8; i++) if (buffer[i] != 0) return false;
    return true;
}

static int read_direct_window(const Record *record, const Shard *shard,
                              unsigned char *buffer, size_t buffer_bytes,
                              uint64_t offset, uint64_t length, const char *range,
                              char *error, size_t error_bytes) {
    uint64_t base = offset & ~(uint64_t)(ALIGNMENT - 1);
    size_t pad = (size_t)(offset - base);
    if (length > SIZE_MAX - pad) goto invalid;
    size_t wanted = pad + (size_t)length;
    size_t direct_length = round_up(wanted, ALIGNMENT);
    if (direct_length > buffer_bytes || base > shard->size) goto invalid;
    uint64_t available = shard->size - base;
    if ((uint64_t)direct_length > available)
        direct_length = (size_t)(available & ~(uint64_t)(ALIGNMENT - 1));
    size_t done = 0;
    while (done < direct_length) {
        ssize_t count = pread(shard->fd, buffer + done, direct_length - done,
                              (off_t)(base + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) goto short_read;
        done += (size_t)count;
    }
    while (done < wanted) {
        ssize_t count = pread(shard->tail_fd, buffer + done, wanted - done,
                              (off_t)(base + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) goto short_read;
        done += (size_t)count;
    }
    memmove(buffer, buffer + pad, (size_t)length);
    if (first_eight_zero(buffer, (size_t)length)) {
        snprintf(error, error_bytes, "all-zero first 8 bytes %s layer=%d expert=%d",
                 range, record->layer, record->expert);
        return -1;
    }
    return 0;

short_read:
    snprintf(error, error_bytes, "short %s read layer=%d expert=%d",
             range, record->layer, record->expert);
    return -1;
invalid:
    snprintf(error, error_bytes, "invalid direct window %s layer=%d expert=%d",
             range, record->layer, record->expert);
    return -1;
}

static void pool_fail(Pool *pool, const char *format, ...) {
    bool expected = false;
    if (!atomic_compare_exchange_strong(&pool->failed, &expected, true)) return;
    va_list args;
    va_start(args, format);
    vsnprintf(pool->error, sizeof(pool->error), format, args);
    va_end(args);
}

static void *worker_main(void *opaque) {
    Worker *worker = opaque;
    Pool *pool = worker->pool;
    unsigned seen = 0;
    pthread_mutex_lock(&pool->mutex);
    while (!pool->stop) {
        while (!pool->stop && seen == pool->generation)
            pthread_cond_wait(&pool->start, &pool->mutex);
        if (pool->stop) break;
        seen = pool->generation;
        pthread_mutex_unlock(&pool->mutex);
        for (;;) {
            if (atomic_load(&pool->failed)) break;
            size_t index = atomic_fetch_add(&pool->next, 1);
            if (index >= pool->work_count) break;
            const Record *record = pool->work[index];
            double started = now_seconds();
            char read_error[512] = {0};
            if (record->shard >= pool->shards->count) {
                pool_fail(pool, "record has invalid shard layer=%d expert=%d",
                          record->layer, record->expert);
                break;
            }
            const Shard *shard = &pool->shards->items[record->shard];
            int result;
            if (record->scale_offset + record->scale_len == record->weight_offset) {
                result = read_direct_window(record, shard, worker->buffer,
                                            pool->buffer_bytes, record->scale_offset,
                                            record->scale_len + record->weight_len,
                                            "record", read_error, sizeof(read_error));
            } else {
                result = read_direct_window(record, shard, worker->buffer,
                                            pool->buffer_bytes, record->weight_offset,
                                            record->weight_len, "weight", read_error,
                                            sizeof(read_error));
                if (!result)
                    result = read_direct_window(record, shard, worker->buffer,
                                                pool->buffer_bytes, record->scale_offset,
                                                record->scale_len, "scale", read_error,
                                                sizeof(read_error));
            }
            if (result) {
                pool_fail(pool, "%s", read_error);
                break;
            }
            pool->latencies_ms[index] = (now_seconds() - started) * 1000.0;
        }
        pthread_mutex_lock(&pool->mutex);
        if (--pool->active == 0) pthread_cond_signal(&pool->done);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static Pool *pool_create(int worker_count, size_t buffer_bytes, const ShardList *shards) {
    Pool *pool = xmalloc(sizeof(*pool));
    memset(pool, 0, sizeof(*pool));
    pool->worker_count = worker_count;
    pool->buffer_bytes = buffer_bytes;
    pool->shards = shards;
    pool->workers = xmalloc((size_t)worker_count * sizeof(*pool->workers));
    pool->threads = xmalloc((size_t)worker_count * sizeof(*pool->threads));
    if (pthread_mutex_init(&pool->mutex, NULL) || pthread_cond_init(&pool->start, NULL) ||
        pthread_cond_init(&pool->done, NULL)) die("pthread synchronization setup failed");
    for (int i = 0; i < worker_count; i++) {
        pool->workers[i].pool = pool;
        if (posix_memalign((void **)&pool->workers[i].buffer, BUFFER_ALIGNMENT,
                           buffer_bytes) != 0)
            die("posix_memalign(%zu) failed", buffer_bytes);
        if (pthread_create(&pool->threads[i], NULL, worker_main, &pool->workers[i]) != 0)
            die("pthread_create failed");
    }
    return pool;
}

static void pool_destroy(Pool *pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->stop = true;
    pthread_cond_broadcast(&pool->start);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->worker_count; i++) {
        pthread_join(pool->threads[i], NULL);
        free(pool->workers[i].buffer);
    }
    pthread_cond_destroy(&pool->done);
    pthread_cond_destroy(&pool->start);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->threads);
    free(pool->workers);
    free(pool);
}

static RoundStats pool_run(Pool *pool, const Record *const *work, size_t count) {
    double *latencies = xmalloc(count * sizeof(*latencies));
    atomic_store(&pool->next, 0);
    atomic_store(&pool->failed, false);
    pool->error[0] = '\0';
    pool->work = work;
    pool->work_count = count;
    pool->latencies_ms = latencies;
    pthread_mutex_lock(&pool->mutex);
    pool->active = pool->worker_count;
    pool->generation++;
    double started = now_seconds();
    pthread_cond_broadcast(&pool->start);
    while (pool->active) pthread_cond_wait(&pool->done, &pool->mutex);
    double elapsed = now_seconds() - started;
    pthread_mutex_unlock(&pool->mutex);
    if (atomic_load(&pool->failed)) die("%s", pool->error);
    qsort(latencies, count, sizeof(*latencies), compare_double);
    size_t p50_index = (count - 1) / 2;
    size_t p99_index = (count * 99 + 99) / 100 - 1;
    uint64_t bytes = 0;
    for (size_t i = 0; i < count; i++) {
        if (work[i]->weight_len > UINT64_MAX - work[i]->scale_len ||
            bytes > UINT64_MAX - work[i]->weight_len - work[i]->scale_len)
            die("byte counter overflow");
        bytes += work[i]->weight_len + work[i]->scale_len;
    }
    RoundStats stats = {.gbps = (double)bytes / elapsed / 1e9,
                        .p50_ms = latencies[p50_index], .p99_ms = latencies[p99_index]};
    free(latencies);
    return stats;
}

static double median(double *values, size_t count) {
    qsort(values, count, sizeof(*values), compare_double);
    return values[count / 2];
}

static RoundStats run_qd(const Record *const *selected, size_t count, int qd,
                         int timed_rounds, const ShardList *shards) {
    const Record **work = xmalloc(count * sizeof(*work));
    memcpy(work, selected, count * sizeof(*work));
    shuffle_records(work, count, SELECT_SEED ^ ((uint64_t)qd << 32));
    Pool *pool = pool_create(qd, max_buffer_bytes(work, count), shards);
    (void)pool_run(pool, work, count);
    double gbps[TIMED_ROUNDS];
    double p50[TIMED_ROUNDS];
    double p99[TIMED_ROUNDS];
    for (int round = 0; round < timed_rounds; round++) {
        RoundStats stats = pool_run(pool, work, count);
        gbps[round] = stats.gbps;
        p50[round] = stats.p50_ms;
        p99[round] = stats.p99_ms;
    }
    pool_destroy(pool);
    free(work);
    return (RoundStats){.gbps = median(gbps, (size_t)timed_rounds),
                        .p50_ms = median(p50, (size_t)timed_rounds),
                        .p99_ms = median(p99, (size_t)timed_rounds)};
}

int main(int argc, char **argv) {
    if (argc > 3) die("usage: %s [model_dir] [layer_contig.json]", argv[0]);
    const char *model_dir = argc > 1 ? argv[1] : "models/deepseek-v4-flash";
    const char *artifact = argc > 2 ? argv[2] : "artifacts/layer_contig.json";
    bool smoke = getenv("QD_SWEEP_SMOKE") && strcmp(getenv("QD_SWEEP_SMOKE"), "1") == 0;
    ShardList shards = {0};
    RecordList records = {0};
    if (access(artifact, R_OK) == 0) {
        load_artifact(artifact, model_dir, &shards, &records);
    } else {
        if (errno != ENOENT) die("cannot read artifact %s: %s", artifact, strerror(errno));
        scan_shard_directory(model_dir, &shards);
        extract_records(&shards, &records);
    }
    require_unique_records(&records);
    size_t work_count = smoke ? SMOKE_RECORDS : DEFAULT_RECORDS;
    const Record **selected = select_work(&records, work_count, !smoke);
    open_needed_shards(&shards, selected, work_count);
    if (smoke) {
        RoundStats stats = run_qd(selected, work_count, 1, 1, &shards);
        printf("QD1: %.3f GB/s p50=%.3fms p99=%.3fms\n",
               stats.gbps, stats.p50_ms, stats.p99_ms);
        printf("SMOKE: QD1 only; GATE_A not evaluated\n");
    } else {
        static const int qds[] = {1, 4, 8, 16, 32};
        RoundStats results[sizeof(qds) / sizeof(qds[0])];
        for (size_t i = 0; i < sizeof(qds) / sizeof(qds[0]); i++) {
            results[i] = run_qd(selected, work_count, qds[i], TIMED_ROUNDS, &shards);
            printf("QD%d: %.3f GB/s p50=%.3fms p99=%.3fms\n", qds[i],
                   results[i].gbps, results[i].p50_ms, results[i].p99_ms);
        }
        double ratio = results[2].gbps / results[0].gbps;
        printf("GATE_A: %s ratio_qd8_qd1=%.2f\n", ratio >= 2.0 ? "PROCEED" : "STOP", ratio);
    }
    free(selected);
    free(records.items);
    close_shards(&shards);
    return EXIT_SUCCESS;
}
