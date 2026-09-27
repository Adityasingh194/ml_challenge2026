"""Execute notebook cells one at a time in a persistent kernel.
  start:  python run_notebook.py start <conn.json>            (launches a kernel, prints its pid)
  run:    python run_notebook.py run <conn.json> <nb.ipynb> <i> [<j> ...]   (code-cell indices; 'code:<python>' runs raw code)
Output is streamed to stdout; a failing cell stops the run with exit code 1."""
import json
import subprocess
import sys
import time

from jupyter_client import BlockingKernelClient


def start(conn):
    p = subprocess.Popen([sys.executable, "-m", "ipykernel_launcher", "-f", conn],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        try:
            json.load(open(conn))
            break
        except Exception:
            time.sleep(0.2)
    print("kernel pid", p.pid)


def run(conn, nb_path, targets):
    kc = BlockingKernelClient()
    kc.load_connection_file(conn)
    kc.start_channels()
    kc.wait_for_ready(timeout=60)
    cells = json.load(open(nb_path))["cells"] if nb_path != "-" else []
    for t in targets:
        if t.startswith("code:"):
            src, label = t[5:], "raw code"
        else:
            c = cells[int(t)]
            if c["cell_type"] != "code":
                continue
            src, label = "".join(c["source"]), f"cell {t}"
        print(f"\n######## {label} ########\n{src.splitlines()[0][:120] if src else ''}", flush=True)
        t0 = time.time()
        msg_id = kc.execute(src)
        failed = False
        while True:
            msg = kc.get_iopub_msg(timeout=None)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            typ, content = msg["msg_type"], msg["content"]
            if typ == "stream":
                sys.stdout.write(content["text"])
                sys.stdout.flush()
            elif typ in ("execute_result", "display_data"):
                print(content["data"].get("text/plain", ""), flush=True)
            elif typ == "error":
                failed = True
                print("\n".join(content["traceback"]), flush=True)
            elif typ == "status" and content["execution_state"] == "idle":
                break
        print(f"######## {label} {'FAILED' if failed else 'ok'} in {time.time() - t0:.1f}s ########", flush=True)
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "start":
        start(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4:])
