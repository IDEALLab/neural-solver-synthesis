"""
Problem catalogs for different domains.
These are the standard problem definitions used in the existing system.
"""

# EPS Catalog - component options for satellite design
EPS_CATALOG = {
    "Orbit": ["LEO-400-DD", "LEO-500-DD", "MEO-1000-DD"],
    "Solar Array": ["XTE-SF", "XTE-LILT", "XTE-HF", "XTJ-CIC", "UTJ-CIC", "XTJ-Prime", "Azur 3G30C"],
    "Battery": ["Saft 8s4p", "Saft 11s16p", "Saft 4s1p VES16",
                "EaglePicher SAR-10197", "EaglePicher SAR-10199",
                "EaglePicher SAR-10207", "EaglePicher SAR-10215"],
    "Degrees of Freedom": ["0", "1", "2"],
}

# Knapsack Catalog - standard item set for knapsack problems
KNAPSACK_CATALOG = [
    {"id": "it-000", "weight": 12, "volume": 8, "value": 25},
    {"id": "it-001", "weight": 18, "volume": 12, "value": 35},
    {"id": "it-002", "weight": 8, "volume": 6, "value": 20},
    {"id": "it-003", "weight": 25, "volume": 15, "value": 45},
    {"id": "it-004", "weight": 15, "volume": 10, "value": 30},
    {"id": "it-005", "weight": 22, "volume": 14, "value": 40},
    {"id": "it-006", "weight": 10, "volume": 7, "value": 22},
    {"id": "it-007", "weight": 28, "volume": 18, "value": 50},
    {"id": "it-008", "weight": 14, "volume": 9, "value": 28},
    {"id": "it-009", "weight": 20, "volume": 13, "value": 38},
    {"id": "it-010", "weight": 16, "volume": 11, "value": 32},
    {"id": "it-011", "weight": 24, "volume": 16, "value": 42},
    {"id": "it-012", "weight": 11, "volume": 8, "value": 24},
    {"id": "it-013", "weight": 19, "volume": 12, "value": 36},
    {"id": "it-014", "weight": 13, "volume": 9, "value": 26},
    {"id": "it-015", "weight": 26, "volume": 17, "value": 48},
    {"id": "it-016", "weight": 17, "volume": 11, "value": 34},
    {"id": "it-017", "weight": 21, "volume": 14, "value": 39},
    {"id": "it-018", "weight": 9, "volume": 6, "value": 18},
    {"id": "it-019", "weight": 23, "volume": 15, "value": 41},
    {"id": "it-020", "weight": 7, "volume": 5, "value": 16},
    {"id": "it-021", "weight": 27, "volume": 18, "value": 49},
    {"id": "it-022", "weight": 12, "volume": 8, "value": 25},
    {"id": "it-023", "weight": 18, "volume": 12, "value": 35},
    {"id": "it-024", "weight": 15, "volume": 10, "value": 30},
    {"id": "it-025", "weight": 20, "volume": 13, "value": 38},
    {"id": "it-026", "weight": 14, "volume": 9, "value": 28},
    {"id": "it-027", "weight": 22, "volume": 14, "value": 40},
    {"id": "it-028", "weight": 16, "volume": 11, "value": 32},
    {"id": "it-029", "weight": 24, "volume": 16, "value": 42},
    {"id": "it-030", "weight": 10, "volume": 7, "value": 22},
    {"id": "it-031", "weight": 19, "volume": 12, "value": 36},
    {"id": "it-032", "weight": 13, "volume": 9, "value": 26},
    {"id": "it-033", "weight": 25, "volume": 15, "value": 45},
    {"id": "it-034", "weight": 17, "volume": 11, "value": 34},
    {"id": "it-035", "weight": 21, "volume": 14, "value": 39},
    {"id": "it-036", "weight": 11, "volume": 8, "value": 24},
    {"id": "it-037", "weight": 26, "volume": 17, "value": 48},
    {"id": "it-038", "weight": 8, "volume": 6, "value": 20},
    {"id": "it-039", "weight": 23, "volume": 15, "value": 41},
    {"id": "it-040", "weight": 9, "volume": 6, "value": 18},
    {"id": "it-041", "weight": 27, "volume": 18, "value": 49},
    {"id": "it-042", "weight": 12, "volume": 8, "value": 25},
    {"id": "it-043", "weight": 18, "volume": 12, "value": 35},
    {"id": "it-044", "weight": 15, "volume": 10, "value": 30},
    {"id": "it-045", "weight": 20, "volume": 13, "value": 38},
    {"id": "it-046", "weight": 14, "volume": 9, "value": 28},
    {"id": "it-047", "weight": 22, "volume": 14, "value": 40},
    {"id": "it-048", "weight": 16, "volume": 11, "value": 32},
    {"id": "it-049", "weight": 24, "volume": 16, "value": 42}
]

# Beams2D Catalog - standard problem parameters
BEAMS2D_CATALOG = {
    "default_requirements": {
        "volfrac": 0.4,
        "rmin": 2.0,
        "forcedist": 0.5,
        "overhang_constraint": False,
        "compliance_min": 8.97,
        "compliance_max": 2129.30
    }
}


def get_eps_catalog():
    """Get EPS component catalog."""
    return EPS_CATALOG


def get_knapsack_catalog():
    """Get knapsack items catalog."""
    return KNAPSACK_CATALOG


def get_beams2d_catalog():
    """Get Beams2D default requirements."""
    return BEAMS2D_CATALOG


def get_default_catalog(domain: str):
    """Get default catalog for a domain."""
    if domain == "eps":
        return get_eps_catalog()
    elif domain == "knapsack":
        return get_knapsack_catalog()
    elif domain == "beams2d":
        return get_beams2d_catalog()
    else:
        raise ValueError(f"Unknown domain: {domain}")
