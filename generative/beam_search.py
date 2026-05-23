"""Trie-constrained beam search for T5 (spec §5.6).

We use HuggingFace's ``generate(..., prefix_allowed_tokens_fn=...)``
mechanism, which masks logits at each decode step to only tokens
allowed by the trie at the current beam prefix.

Returns the top-K item ids (deduplicated; multiple beams may reach the
same leaf via different paths if there are identifier collisions).
"""

from __future__ import annotations
import torch

from .trie import ItemTrie


def make_prefix_allowed_tokens_fn(trie: ItemTrie, decoder_start_id: int,
                                    eos_token_id: int | None = None):
    """Build a callback compatible with HF's `prefix_allowed_tokens_fn`.

    The HF callback signature is `fn(batch_id: int, input_ids: Tensor) -> List[int]`.
    `input_ids` already includes the decoder_start token at position 0 (T5
    uses pad_token_id as the BOS for decoding by default), so we strip it
    before traversing the trie.

    When a beam reaches a trie leaf (no more valid next tokens), HF raises
    if we return an empty list, so we instead force the EOS token id so the
    beam terminates gracefully.
    """

    def fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        ids = input_ids.tolist()
        # Drop the leading decoder_start token
        if ids and ids[0] == decoder_start_id:
            ids = ids[1:]
        valid = trie.valid_next_tokens(ids)
        if not valid:
            # Reached a leaf or invalid path — emit EOS so generation stops.
            return [eos_token_id] if eos_token_id is not None else [decoder_start_id]
        return valid

    return fn


@torch.no_grad()
def trie_beam_search(
    t5,                                         # T5ForConditionalGeneration
    inputs_embeds: torch.Tensor,                # (B, S, d_t5)
    attention_mask: torch.Tensor,               # (B, S)
    trie: ItemTrie,
    eoi_token_id: int,                          # custom <EOI>
    beam_width: int = 50,
    num_return_sequences: int = 20,
) -> list[list[int]]:
    """Returns list of length B; each element is a list of top item_ids."""
    decoder_start_id = t5.config.decoder_start_token_id
    if decoder_start_id is None:
        decoder_start_id = t5.config.pad_token_id

    prefix_fn = make_prefix_allowed_tokens_fn(trie, decoder_start_id)
    output = t5.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        num_beams=beam_width,
        num_return_sequences=min(num_return_sequences, beam_width),
        max_new_tokens=trie.max_depth + 2,   # cushion for stops
        prefix_allowed_tokens_fn=prefix_fn,
        eos_token_id=eoi_token_id,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=False,
    )
    sequences = output.sequences  # (B * num_return, T)
    B = inputs_embeds.shape[0]
    R = sequences.shape[0] // B
    seqs = sequences.view(B, R, -1)

    # Map each generated sequence to an item_id via trie traversal.
    results: list[list[int]] = []
    for b in range(B):
        items_for_b: list[int] = []
        seen: set[int] = set()
        for r in range(R):
            ids = seqs[b, r].tolist()
            # Drop leading decoder_start
            if ids and ids[0] == decoder_start_id:
                ids = ids[1:]
            # Truncate at the first EOI (HF may emit EOS afterwards too)
            if eoi_token_id in ids:
                cut = ids.index(eoi_token_id) + 1
                ids = ids[:cut]
            node = trie.get_node(ids)
            if node is not None and node.is_leaf and node.item_id is not None:
                if node.item_id not in seen:
                    items_for_b.append(node.item_id)
                    seen.add(node.item_id)
        results.append(items_for_b)
    return results
