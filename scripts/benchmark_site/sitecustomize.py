"""Benchmark-only Python socket audit, inherited by spawned interpreters."""

import json
import os
import sys
from pathlib import Path

if os.environ.get("HITLIST_BENCHMARK_DENY_NETWORK") == "1":
    directory = Path(os.environ["HITLIST_BENCHMARK_NETWORK_LOG"])
    directory.mkdir(parents=True, exist_ok=True)

    def reject_network(event, arguments):
        denied = event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
        if event in {"socket.connect", "socket.sendto"}:
            # multiprocessing may use local AF_UNIX sockets; Internet sockets
            # carry tuple addresses. Network access is never a build input here.
            denied = not isinstance(arguments[-1], (str, bytes))
        if denied:
            with (directory / f"{os.getpid()}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"pid": os.getpid(), "event": event, "address": repr(arguments[-1])})
                    + "\n"
                )
            raise RuntimeError(f"Timed benchmark attempted network access: {event}")

    sys.addaudithook(reject_network)
    (directory / f"{os.getpid()}.installed").write_text("Python socket audit installed\n")
