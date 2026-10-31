"""The broad species list, and danger groups matched on taxonomy.

Matching on taxonomy, not on name text, means a sea lion is not a lion and a leopard cat
is not a leopard.
"""
import csv
from functools import lru_cache
from pathlib import Path

LIST_PATH = Path(__file__).resolve().parent / "species_list.csv"


@lru_cache(maxsize=1)
def load_species():
    """{common name: {"scientific": ..., "taxonomy": [class, order, family, genus, species]}}"""
    with open(LIST_PATH, newline="", encoding="utf-8") as f:
        return {r["common"]: {"scientific": r["scientific"], "taxonomy": r["taxonomy"].split(";")} for r in csv.DictReader(f)}


def group_for(name, cfg):
    """The first danger group in config whose taxa match this species, or None."""
    species = load_species().get(name)
    if species is None:
        return None
    taxonomy = species["taxonomy"]
    tokens = set(taxonomy) | {species["scientific"]}
    for group, rule in cfg["danger_groups"].items():
        if tokens & set(rule["taxa"]):
            return group
    return None
