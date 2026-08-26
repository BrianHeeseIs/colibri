# `COLI_V4_HOT_PACK_UNLOCKED` design note

## 2026-08-26 hand-port

`COLI_V4_HOT_PACK_UNLOCKED` is read once through `pthread_once` and defaults
off.  A disabled policy has no `pack_mutexes` allocation, and both hot-pack
call sites retain the old path: test pin status and call
`hot_pack_slot_locked()` while `state->mutex` remains held.

When enabled, each cache slot has one pack mutex.  `lookup_hot()` establishes
its lease first at both call sites: a resident hit increments
`slot->references`, while a miss sets `slot->references = 1` before dropping
the store mutex for its read.  The miss does not publish `slot->expert` until
the read has completed under the store mutex.  `release()` can decrement a
reference only for a view already returned to its caller.

`hot_prepare_slot()` enters and returns with `state->mutex` held.  It drops
that mutex before waiting for the slot's pack mutex, reacquires the store mutex,
and permits an in-place conversion only when this lookup still owns the sole
reference and the slot is not already packed.  It then drops only
`state->mutex`, keeps the pack mutex across the unchanged gate/down/up packing
computation, reacquires `state->mutex` to publish `packed[slot]` and increment
`packed_slots`, and releases the pack mutex.  Every enabled lookup crosses the
slot mutex, including a lookup that did not request packing, so no view can be
published while conversion of its slab is in flight.

The caller's not-yet-returned lease protects lifetime while the store lock is
dropped.  Victim selection considers only slots with `references == 0`, so the
slot cannot be evicted or reused during packing.  Its reference also cannot
drop to zero: no caller can release this lookup's view before `lookup_hot()`
returns.  Destruction requires zero active leases before pack mutexes or slabs
are destroyed.

Races excluded by construction:

- Two packers for one slot serialize on its pack mutex and recheck
  `packed[slot]` before conversion.
- A pre-existing row-major reader prevents conversion because the exclusive
  `references == 1` check fails.  A reader arriving after that check waits on
  the pack mutex before publishing its view.
- Lock inversion is avoided: no path waits for a pack mutex while holding
  `state->mutex`; the only nested order is pack mutex, then store mutex.
- Slot reuse cannot race the `packed` flag reset because reuse requires zero
  references, while every prepare/wait path owns one reference.

Packing arithmetic, matrix order, scratch sizing, and in-place memcpy order are
unchanged.  Only lock ownership and packed-state publication moved.
