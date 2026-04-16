"""Upgrade 7 — Entity risk graph for fraud network detection.

Builds a heterogeneous graph connecting: card, merchant, device_id, ip, location.
Propagates risk through shared entities to surface fraud rings.
"""

from __future__ import annotations

import json
from collections import defaultdict
from threading import Lock
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("entity_graph")

# Entity types
CARD = "card"
MERCHANT = "merchant"
DEVICE = "device"
IP = "ip"
LOCATION = "location"


class EntityRiskGraph:
    """Lightweight fraud network graph using adjacency lists.

    Each transaction creates edges between its entities (card, merchant, device, ip, location).
    Risk propagates: if a device is involved in multiple anomalies, all connected entities
    get elevated risk.
    """

    def __init__(self, decay: float = 0.95, propagation_hops: int = 2):
        self._adj: dict[str, set[str]] = defaultdict(set)       # entity -> neighbors
        self._risk: dict[str, float] = defaultdict(float)        # entity -> risk score
        self._txn_count: dict[str, int] = defaultdict(int)       # entity -> txn count
        self._anomaly_count: dict[str, int] = defaultdict(int)   # entity -> anomaly count
        self._entity_type: dict[str, str] = {}                   # entity -> type
        self._txn_entities: list[list[str]] = []                 # per-txn entity list
        self._decay = decay
        self._hops = propagation_hops
        self._lock = Lock()

    def _make_id(self, entity_type: str, value: str) -> str:
        return f"{entity_type}:{value}"

    def add_transaction(
        self,
        card_id: str = "unknown",
        merchant_id: str = "unknown",
        device_id: str = "unknown",
        ip_address: str = "unknown",
        location: str = "unknown",
        is_anomaly: bool = False,
        anomaly_score: float = 0.0,
    ) -> dict:
        """Register a transaction in the graph and return graph-based risk signals."""
        entities = []
        for etype, val in [
            (CARD, card_id), (MERCHANT, merchant_id),
            (DEVICE, device_id), (IP, ip_address), (LOCATION, location),
        ]:
            if val and val != "unknown":
                eid = self._make_id(etype, val)
                entities.append(eid)
                self._entity_type[eid] = etype

        with self._lock:
            # Create edges between all entity pairs in this transaction
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    self._adj[e1].add(e2)
                    self._adj[e2].add(e1)
                self._txn_count[e1] = self._txn_count.get(e1, 0) + 1
                if is_anomaly:
                    self._anomaly_count[e1] = self._anomaly_count.get(e1, 0) + 1
                    # Inject risk at anomaly source
                    self._risk[e1] = min(1.0, self._risk.get(e1, 0) + anomaly_score * 0.3)

            self._txn_entities.append(entities)

            # Propagate risk through network (BFS from anomaly entities)
            if is_anomaly:
                self._propagate_risk(entities)

            # Compute graph signals for this transaction
            signals = self._compute_signals(entities)

        return signals

    def _propagate_risk(self, source_entities: list[str]):
        """BFS risk propagation from source entities up to N hops."""
        visited = set(source_entities)
        frontier = set(source_entities)
        for hop in range(self._hops):
            next_frontier = set()
            for entity in frontier:
                for neighbor in self._adj.get(entity, set()):
                    if neighbor not in visited:
                        # Decayed risk propagation
                        propagated = self._risk.get(entity, 0) * (self._decay ** (hop + 1))
                        self._risk[neighbor] = min(1.0, self._risk.get(neighbor, 0) + propagated * 0.5)
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

    def _compute_signals(self, entities: list[str]) -> dict:
        """Compute graph-based risk signals for a transaction's entities."""
        if not entities:
            return {
                "entity_risk_mean": 0.0, "entity_risk_max": 0.0,
                "risky_neighbors": 0, "shared_device_risk": 0.0,
                "merchant_anomaly_density": 0.0, "cluster_size": 0,
                "entity_count": 0,
            }

        risks = [self._risk.get(e, 0) for e in entities]
        neighbor_risks = []
        for e in entities:
            for n in self._adj.get(e, set()):
                if self._risk.get(n, 0) > 0.3:
                    neighbor_risks.append(self._risk[n])

        # Device-specific risk
        device_entities = [e for e in entities if self._entity_type.get(e) == DEVICE]
        device_risk = max([self._risk.get(e, 0) for e in device_entities], default=0.0)

        # Merchant anomaly density
        merchant_entities = [e for e in entities if self._entity_type.get(e) == MERCHANT]
        merchant_density = 0.0
        for me in merchant_entities:
            tc = self._txn_count.get(me, 1)
            ac = self._anomaly_count.get(me, 0)
            merchant_density = max(merchant_density, ac / tc)

        # Cluster size: total unique entities reachable in 1 hop
        cluster = set()
        for e in entities:
            cluster.add(e)
            cluster.update(self._adj.get(e, set()))

        return {
            "entity_risk_mean": round(float(np.mean(risks)), 4),
            "entity_risk_max": round(float(np.max(risks)), 4),
            "risky_neighbors": len(neighbor_risks),
            "shared_device_risk": round(device_risk, 4),
            "merchant_anomaly_density": round(merchant_density, 4),
            "cluster_size": len(cluster),
            "entity_count": len(entities),
        }

    def get_entity_info(self, entity_id: str) -> dict:
        """Get info about a specific entity node."""
        with self._lock:
            return {
                "entity_id": entity_id,
                "type": self._entity_type.get(entity_id, "unknown"),
                "risk": round(self._risk.get(entity_id, 0), 4),
                "txn_count": self._txn_count.get(entity_id, 0),
                "anomaly_count": self._anomaly_count.get(entity_id, 0),
                "neighbor_count": len(self._adj.get(entity_id, set())),
                "neighbors": list(self._adj.get(entity_id, set()))[:20],
            }

    def get_suspicious_clusters(self, min_risk: float = 0.3, top_n: int = 10) -> list[dict]:
        """Find clusters of connected high-risk entities."""
        with self._lock:
            risky = {e for e, r in self._risk.items() if r >= min_risk}
            if not risky:
                return []

            # BFS to find connected components among risky entities
            visited = set()
            clusters = []
            for start in risky:
                if start in visited:
                    continue
                cluster = set()
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    cluster.add(node)
                    for neighbor in self._adj.get(node, set()):
                        if neighbor in risky and neighbor not in visited:
                            queue.append(neighbor)

                if len(cluster) >= 2:
                    cluster_risk = np.mean([self._risk.get(e, 0) for e in cluster])
                    clusters.append({
                        "entities": list(cluster)[:50],
                        "size": len(cluster),
                        "avg_risk": round(float(cluster_risk), 4),
                        "total_anomalies": sum(self._anomaly_count.get(e, 0) for e in cluster),
                        "entity_types": dict(
                            sorted(
                                defaultdict(int, {
                                    self._entity_type.get(e, "?"): 1
                                    for e in cluster
                                }).items()
                            )
                        ),
                    })

            return sorted(clusters, key=lambda c: c["avg_risk"], reverse=True)[:top_n]

    def get_graph_data_for_viz(self, max_nodes: int = 200) -> dict:
        """Export graph data for Plotly/dashboard visualization."""
        with self._lock:
            # Take top-risk nodes
            sorted_entities = sorted(self._risk.items(), key=lambda x: x[1], reverse=True)
            top_entities = set(e for e, _ in sorted_entities[:max_nodes])

            # Also include their neighbors
            expanded = set(top_entities)
            for e in top_entities:
                for n in list(self._adj.get(e, set()))[:5]:
                    expanded.add(n)
                    if len(expanded) >= max_nodes:
                        break

            nodes = []
            for e in expanded:
                nodes.append({
                    "id": e,
                    "type": self._entity_type.get(e, "unknown"),
                    "risk": round(self._risk.get(e, 0), 4),
                    "txn_count": self._txn_count.get(e, 0),
                    "anomaly_count": self._anomaly_count.get(e, 0),
                })

            edges = []
            seen_edges = set()
            for e in expanded:
                for n in self._adj.get(e, set()):
                    if n in expanded:
                        edge_key = tuple(sorted([e, n]))
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edges.append({"source": e, "target": n})

            return {"nodes": nodes, "edges": edges}

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_entities": len(self._entity_type),
                "total_edges": sum(len(v) for v in self._adj.values()) // 2,
                "risky_entities": sum(1 for r in self._risk.values() if r > 0.3),
                "total_transactions": len(self._txn_entities),
            }
