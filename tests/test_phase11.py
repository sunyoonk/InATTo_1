"""Phase 11: trie + beam search smoke test on a tiny synthetic scenario."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from generative.trie import ItemTrie


def test_trie_basic():
    # 4 items with overlapping identifier prefixes
    idents = [
        [1, 2, 3],       # item 0
        [1, 2, 4],       # item 1 — shares prefix [1, 2]
        [1, 5, 6],       # item 2 — shares first token only
        [9, 9, 9],       # item 3 — completely separate
    ]
    trie = ItemTrie.from_identifiers(idents, item_ids=[0, 1, 2, 3])
    assert trie.n_items == 4
    assert trie.max_depth == 3
    assert sorted(trie.valid_next_tokens([])) == [1, 9]
    assert sorted(trie.valid_next_tokens([1])) == [2, 5]
    assert sorted(trie.valid_next_tokens([1, 2])) == [3, 4]
    # Leaf lookup
    node = trie.get_node([1, 2, 3])
    assert node is not None and node.is_leaf and node.item_id == 0
    node = trie.get_node([9, 9])
    assert node is not None and not node.is_leaf
    # Out-of-vocab path
    assert trie.get_node([7, 8]) is None
    print("[OK] trie: insert / valid_next / leaf lookup all work")


def test_trie_collision():
    idents = [[1, 2], [1, 2], [3]]   # items 0 and 1 collide
    trie = ItemTrie.from_identifiers(idents, item_ids=[0, 1, 2])
    n_coll = trie.collision_count()
    assert n_coll == 2, f"expected 2 colliding items, got {n_coll}"
    print(f"[OK] trie collision count = {n_coll}")


if __name__ == "__main__":
    test_trie_basic()
    test_trie_collision()
    print("\nALL PHASE 11 TESTS PASSED ✓")
