### Checklist

- [ ] If this is not a feature request but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [ ] Please use English. Otherwise, it will be closed.

### Motivation

I'd briefly share my plan of action on how we can support Speculative Decoding + PP in Disagg Decode Mode. Feel free to drop your comments/suggestions.

## Model Weights Layout 
- The speculative model weights will be duplicated loaded on `first_pp_rank` and `last_pp_rank`:
  - This implies duplicate `lm_head` on the `first_pp_rank` and duplicate `embed` on the `last_pp_rank`. 
- The target model weights will be set-up in the regular PP setting.

## Flow
- Draft Phase would happen on the `first_pp_rank` -> `draft_tokens`, `scores` would be sent to each PP-rank via the `PPProxyTensors` already implemented.
- Each individual PP-rank would materialize `build_tree_kernel_efficient()` to get `EagleVerifyInput`.
- The last PP-rank would run the `verify()` to get the `accepted_tokens` -> `verify_metadata` sent to the `first_pp_rank`.
- The last PP-rank would also run `forward_draft_extend_after_decode()`.
- `_prep_batch_result` would run the `post_process_after_verify(verify_metadata)` to update request states (finished/unfinished) and free unused KV-slots on the first PP-rank.
- `verify_metadata` would then be passed along to PP-1..PP-2 and so on for request state adjustment on each PP-rank.

## Diagram
## Speculative Decoding + Pipeline Parallelism in Disagg Decode Mode (Flow of a Microbatch)

```mermaid
sequenceDiagram
    participant C as Client
    participant PP0 as first_pp_rank<br/>(PP0)
    participant PPn as middle_pp_ranks<br/>(PP-1..PP-N-2)
    participant PPL as last_pp_rank<br/>(PP-N-1)

    note over PP0: Hosts: embed + mtp_layer + lm_head<br/>+ target model (first PP shard)
    note over PPL: Hosts: embed + mtp_layer + lm_head<br/>+ target model (last PP shard)
    note over PPn: Hosts: target model (middle PP shard)

    C->>PP0: Decode Request

    note over PP0: Draft Phase
    PP0->>PP0: draft() → EagleDraftOutput

    note over PP0: Tree Building
    PP0->>PP0: build_tree_kernel_efficient() → EagleVerifyInput

    note over PP0: Target Forward
    PP0->>PP0: forward() -> hidden_states

    PP0->>PPn: PPProxyTensors(hidden_states, EagleDraftOutput)

    note over PPn: Tree Building
    PPn->>PPn: build_tree_kernel_efficient() → EagleVerifyInput

    note over PPn: Target Forward
    PPn->>PPn: forward() -> hidden_states

    PPn->>PPL: PPProxyTensors(hidden_states, EagleDraftOutput)

    note over PPL: Tree Building
    PPL->>PPL: build_tree_kernel_efficient() → EagleVerifyInput

    note over PPL: Target Forward + Verify
    PPL->>PPL: foward() → hidden_states, logits
    PPL->>PPL: verify() -> verify_metadata[accepted_tokens, pages_to_free,...]

    note over PPL: Draft Extend
    PPL->>PPL: forward_draft_extend_after_decode()
    PPL-->>PP0: verify_metadata

    note over PP0: Post-Process
    PP0->>PP0: _prep_batch_result()<br/>post_process_after_verify(verify_metadata)<br/>→ update request states, free KV-slots
```


## Target Worker
- I plan to extend `EagleWorker` module for this support (since PP doesn't support overlap mode anyways so there's no point working with `EagleWorker2`.


## Concerns
- Since `draft()` happens on first PP-rank and the `draft_extend_after_decode()` happens on last PP-rank, there is a necessity to pass `incremental KVCache` across PP-ranks -> Need some feedback on the feasibility of doing this.

### Related resources

_No response_
