"""Schema validators for mixed SDS/JSSP/CVRP problem records."""

from __future__ import annotations

from typing import Any

SUPPORTED_DOMAINS = {"sds", "jssp", "cvrp", "tsp"}


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_jssp_requirements(requirements: dict[str, Any]) -> bool:
    """Validate JSSP requirements schema."""
    if not isinstance(requirements, dict):
        return False
    if not isinstance(requirements.get("n_jobs"), int):
        return False
    if not isinstance(requirements.get("n_machines"), int):
        return False
    jobs = requirements.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != requirements["n_jobs"]:
        return False
    for job in jobs:
        if not isinstance(job, dict):
            return False
        if not isinstance(job.get("job_id"), int):
            return False
        ops = job.get("operations")
        if not isinstance(ops, list) or len(ops) == 0:
            return False
        for op in ops:
            if not isinstance(op, dict):
                return False
            if not isinstance(op.get("op_id"), int):
                return False
            if not isinstance(op.get("machine_id"), int):
                return False
            if not isinstance(op.get("proc_time"), int):
                return False
    return True


def validate_cvrp_requirements(requirements: dict[str, Any]) -> bool:
    """Validate CVRP requirements schema."""
    if not isinstance(requirements, dict):
        return False
    int_keys = {"n_customers", "n_vehicles", "vehicle_capacity"}
    if any(not isinstance(requirements.get(key), int) for key in int_keys):
        return False
    if requirements["n_customers"] <= 0 or requirements["n_vehicles"] <= 0:
        return False
    depot = requirements.get("depot")
    if not isinstance(depot, dict):
        return False
    if "x" not in depot or "y" not in depot:
        return False
    customers = requirements.get("customers")
    if not isinstance(customers, list) or len(customers) != requirements["n_customers"]:
        return False
    for customer in customers:
        if not isinstance(customer, dict):
            return False
        if not isinstance(customer.get("id"), int):
            return False
        if not isinstance(customer.get("demand"), int):
            return False
        if "x" not in customer or "y" not in customer:
            return False
    return True


def validate_tsp_requirements(requirements: dict[str, Any]) -> bool:
    """Validate the strict synthetic TSP requirements schema."""
    if not isinstance(requirements, dict):
        return False
    n_cities = requirements.get("n_cities")
    cities = requirements.get("cities")
    if not isinstance(n_cities, int) or n_cities < 3:
        return False
    if requirements.get("edge_weight_type") != "EUC_2D":
        return False
    if not isinstance(cities, list) or len(cities) != n_cities:
        return False
    coordinates = set()
    for expected_id, city in enumerate(cities):
        if not isinstance(city, dict) or city.get("id") != expected_id:
            return False
        if isinstance(city.get("x"), bool) or not isinstance(city.get("x"), int):
            return False
        if isinstance(city.get("y"), bool) or not isinstance(city.get("y"), int):
            return False
        coordinates.add((city["x"], city["y"]))
    return len(coordinates) == n_cities


def validate_problem_record(record: dict[str, Any]) -> bool:
    """Validate a generic problem record from the mixed suite."""
    if not isinstance(record, dict):
        return False
    required_keys = {
        "uuid",
        "domain",
        "problem_type",
        "mission",
        "requirements",
        "catalog",
        "target",
    }
    if not required_keys.issubset(record):
        return False
    if not _is_non_empty_str(record.get("uuid")):
        return False
    domain = record.get("domain")
    if domain not in SUPPORTED_DOMAINS:
        return False
    if not isinstance(record.get("mission"), dict):
        return False
    if not isinstance(record.get("requirements"), dict):
        return False
    if not isinstance(record.get("catalog"), dict):
        return False
    if not isinstance(record.get("target"), dict):
        return False

    if domain == "jssp":
        return validate_jssp_requirements(record["requirements"])
    if domain == "cvrp":
        return validate_cvrp_requirements(record["requirements"])
    if domain == "tsp":
        return validate_tsp_requirements(record["requirements"])
    return True


def validate_prompt_record(record: dict[str, Any]) -> bool:
    """Validate a rendered prompt record."""
    if not isinstance(record, dict):
        return False
    required_keys = {"uuid", "problem", "mission", "domain"}
    if not required_keys.issubset(record):
        return False
    if not _is_non_empty_str(record.get("uuid")):
        return False
    if not _is_non_empty_str(record.get("problem")):
        return False
    if not isinstance(record.get("mission"), dict):
        return False
    if record.get("domain") not in SUPPORTED_DOMAINS:
        return False
    return True
