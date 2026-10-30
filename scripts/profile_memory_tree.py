"""Bounded procfs/child-self RSS reconciliation for benchmark issue #876."""

import json
import os
from pathlib import Path

import modal

if modal.is_local():
    from reliability_modal import image
else:
    image = None

app = modal.App("hitlist-memory-diagnostic")


@app.function(image=image, cpu=(8, 8), memory=(2048, 2048), timeout=120, retries=0)
def probe():
    import contextlib
    import subprocess
    import sys

    sys.path.insert(0, "/opt/benchmark")
    from profile_build import MemorySampler

    def snapshot(pid):
        path = Path(f"/proc/{pid}")
        status = dict(
            line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line
        )
        return {
            "pid": pid,
            "page_bytes": os.sysconf("SC_PAGE_SIZE"),
            "statm": (path / "statm").read_text().strip(),
            "status": {
                key: status.get(key, "").strip()
                for key in (
                    "Name",
                    "Tgid",
                    "Pid",
                    "PPid",
                    "Threads",
                    "VmRSS",
                    "RssAnon",
                    "RssFile",
                    "RssShmem",
                )
            },
            "children": (path / f"task/{pid}/children").read_text().split(),
        }

    code = """import json,os,resource,threading,time
from pathlib import Path
buffer=bytearray(128*1024**2)
threads=[threading.Thread(target=time.sleep,args=(30,),daemon=True) for _ in range(4)]
for thread in threads: thread.start()
pid=os.getpid()
print(json.dumps({'pid':pid,'page_bytes':os.sysconf('SC_PAGE_SIZE'),'statm':Path('/proc/self/statm').read_text().strip(),'status':Path('/proc/self/status').read_text(),'own_statm':Path(f'/proc/{pid}/statm').read_text().strip(),'own_status':Path(f'/proc/{pid}/status').read_text(),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
time.sleep(30)
"""
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        child = json.loads(process.stdout.readline())
        sampler = MemorySampler()
        sampler.sample()
        pending, seen, tree = [os.getpid()], set(), []
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                info = snapshot(pid)
            except (OSError, ValueError):
                continue
            tree.append(info)
            pending.extend(map(int, info["children"]))
        return {
            "sampler": sampler.total,
            "child_self": child,
            "tree": tree,
            "platform": sys.platform,
        }
    finally:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


@app.local_entrypoint()
def main(output: str):
    result = probe.remote()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
