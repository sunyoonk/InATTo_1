"""Item identifier trie for constrained beam search (spec §5.6).

Built once from the (T5-token-id) identifier sequence of every item in
the training catalogue. Leaf nodes are at variable depths corresponding
to each item's active identifier length.

Beam search consults the trie at every step to mask out invalid next-
tokens, so generation cannot diverge from the item vocabulary.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    is_leaf: bool = False
    item_id: Optional[int] = None


class ItemTrie:
    def __init__(self):
        self.root = TrieNode()
        self.n_items: int = 0
        self.max_depth: int = 0
        # Reverse map: item_id -> identifier (for analysis/eval)
        self.item_to_identifier: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_identifiers(
        cls,
        identifiers: Iterable[list[int]],
        item_ids: Iterable[int] | None = None,
    ) -> "ItemTrie":
        trie = cls()
        if item_ids is None:
            item_ids = range(sum(1 for _ in identifiers))
        # We need to iterate twice; materialize.
        ids = list(item_ids)
        idents = list(identifiers)
        for iid, ident in zip(ids, idents):
            trie.insert(ident, iid)
        return trie

    def insert(self, identifier: list[int], item_id: int):
        node = self.root
        for tok in identifier:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        if node.is_leaf and node.item_id != item_id:
            # Collision: two items share the same identifier sequence.
            # We disambiguate by overwriting (last-write-wins) and counting.
            # Caller should track collisions externally.
            pass
        node.is_leaf = True
        node.item_id = item_id
        self.item_to_identifier[item_id] = identifier
        self.n_items += 1
        self.max_depth = max(self.max_depth, len(identifier))

    # ------------------------------------------------------------------
    def get_node(self, prefix: list[int]) -> Optional[TrieNode]:
        node = self.root
        for tok in prefix:
            if tok not in node.children:
                return None
            node = node.children[tok]
        return node

    def valid_next_tokens(self, prefix: list[int]) -> list[int]:
        node = self.get_node(prefix)
        if node is None:
            return []
        return list(node.children.keys())

    def collision_count(self) -> int:
        """How many items share an identifier with another item."""
        seen: dict[tuple[int, ...], list[int]] = {}
        for iid, ident in self.item_to_identifier.items():
            seen.setdefault(tuple(ident), []).append(iid)
        return sum(len(v) for v in seen.values() if len(v) > 1)
