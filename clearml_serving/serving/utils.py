from typing import List, Set

import grpc


def parse_grpc_errors(errors: List[str]) -> Set[grpc.StatusCode]:
    try:
        typed_errors = {
            int(e) if e.isdigit() else e.lower().replace("-", " ").replace("_", " ")
            for e in errors
        }
        if len(typed_errors) == 1 and next(iter(typed_errors)) in ("true", "false"):
            return set(grpc.StatusCode if next(iter(typed_errors)) == "true" else [])
        return {e for e in grpc.StatusCode if typed_errors.intersection(e.value)}
    except (ValueError, TypeError):
        pass
    return set()
