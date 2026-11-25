"""Campaign runner: run the jobs of one or more experiments (``experiments.py``) on one machine.

    PY=.venv/bin/python
    $PY scripts/campaign/run.py --root results/campaign --src results/campaign/code --exp S2_screen --dry-run
    nohup $PY scripts/campaign/run.py --root results/campaign --src results/campaign/code \\
        --exp H1_starts H2_scale --slots 7 --mem-gb 56 > results/campaign/nohup_S3.out 2>&1 &
    tail -f results/campaign/progress.txt

Layout (everything under ``--root``; the folder can be copied as a whole, links are relative):

    instances/<I>/                      instances.py
    starts/<S>/<I>/                     starts.py
    runs/<exp>/<arm>/<S>/precalculated_instances/<I>/   job folder: links to the instance and start,
                                        config.json, experiments/<timestamp>/ (matheuristic output) or
                                        cp_mono/seed<k>/ (cp_mono.py output)
    runs/<exp>/_done/<arm>__<S>__<I>.json    one per finished job (rc, wall time, peak RSS, command)
    runs/<exp>/logs/<arm>__<S>__<I>.{log,err}
    progress.txt                        one line per finished job with an ETA

Scheduling
- At most ``--slots`` jobs at a time, each pinned to its own core from ``--cores`` (``taskset``; check
  ``lscpu -e`` that these are distinct *physical* cores, on the Ryzen 7700X cpus 0-7 usually are), and the
  sum of the jobs' memory estimates (``experiments.MEM_GB``) kept below ``--mem-gb``. Jobs are started
  longest-first by estimated duration; a job that does not fit in the free memory waits and the next one
  that fits starts instead.
- ``--cgroup`` additionally runs every job under ``systemd-run --user --scope -p MemoryMax=<1.5 x estimate>``,
  so a job that blows up is killed alone instead of the kernel OOM killer picking any process.
- Time-limited comparisons are only fair at a constant load: run one stage at a time with the same
  ``--slots``, and never split one experiment over machines.
- Finished jobs (``_done`` marker with rc 0) are skipped, so the runner can be stopped (Ctrl-C / SIGTERM kills
  the running jobs; they are re-run next time) and restarted. A failed job is retried on the next start
  unless ``--skip-failed``.

Code: jobs run ``<src>/matheuristic.py`` / ``<src>/scripts/campaign/cp_mono.py`` from a frozen copy of the repo
(``freeze.sh``), so the working tree can change while the campaign runs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from experiments import EXPERIMENTS, Experiment  # noqa: E402


def flags(d: dict) -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            out.append(f"--{k}" if v else f"--no-{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def shape(instance: str) -> tuple[int, int]:
    parts = instance.split("-")
    return int(parts[1]), int(parts[4])


class Job:
    def __init__(self, exp: Experiment, arm: str, start: str, instance: str, root: str):
        self.exp, self.arm, self.start, self.instance = exp, arm, start, instance
        self.id = f"{arm}__{start}__{instance}"
        self.exp_dir = os.path.join(root, "runs", exp.name)
        self.cwd = os.path.join(self.exp_dir, arm, start)
        self.folder = os.path.join(self.cwd, "precalculated_instances", instance)
        self.done_file = os.path.join(self.exp_dir, "_done", self.id + ".json")
        self.mem = exp.mem_gb(arm, instance)
        self.est = exp.est_seconds(arm, instance)
        self.cp = exp.model(arm) == "cp"

    def status(self) -> str | None:
        if not os.path.exists(self.done_file):
            return None
        with open(self.done_file) as f:
            return "ok" if json.load(f)["rc"] == 0 else "failed"

    def prepare(self, root: str) -> None:
        inst_src = os.path.join(root, "instances", self.instance)
        start_src = os.path.join(root, "starts", self.start, self.instance)
        for p in (os.path.join(inst_src, "instance.pkl"), os.path.join(start_src, "heuristic_solution.pkl")):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{self.id}: missing {p} (run instances.py / starts.py)")
        os.makedirs(os.path.join(self.folder, "experiments"), exist_ok=True)
        for src_dir, names in ((inst_src, ("instance.pkl", "instance_summary.txt")),
                               (start_src, ("heuristic_solution.pkl", "heuristic_summary.txt"))):
            for fn in names:
                link = os.path.join(self.folder, fn)
                if not os.path.lexists(link):
                    os.symlink(os.path.relpath(os.path.join(src_dir, fn), self.folder), link)
        with open(os.path.join(inst_src, "config.json")) as f:
            config = json.load(f)
        config["heuristic"] = self.start
        with open(os.path.join(self.folder, "config.json"), "w") as f:
            json.dump(config, f)

    def commands(self, py: str, src: str) -> list[list[str]]:
        if self.cp:
            return [[py, os.path.join(src, "scripts", "campaign", "cp_mono.py"), self.folder,
                     "--time-limit", str(self.exp.time_limit), "--seed", str(seed)] for seed in self.exp.seeds]
        budget = ["--time-limit", str(self.exp.time_limit)] if self.exp.time_limit else ["--iters", str(self.exp.iters)]
        return [[py, os.path.join(src, "matheuristic.py"), self.instance, "--runs", str(self.exp.runs), *budget,
                 *flags(self.exp.arms[self.arm])]]


def build_jobs(root: str, names: list[str], arm_glob: str, start_glob: str, inst_glob: str) -> list[Job]:
    with open(os.path.join(root, "instances", "sets.json")) as f:
        sets = json.load(f)
    jobs = []
    for name in names:
        exp = EXPERIMENTS[name]
        insts = [i for s in exp.sets for i in sets[s]]
        if exp.shapes:
            insts = [i for i in insts if shape(i) in exp.shapes]
        insts = [i for i in insts if fnmatch.fnmatch(i, exp.instance_glob)]
        for arm in exp.arms:
            for start in exp.starts:
                for inst in insts:
                    if fnmatch.fnmatch(arm, arm_glob) and fnmatch.fnmatch(start, start_glob) \
                            and fnmatch.fnmatch(inst, inst_glob):
                        jobs.append(Job(exp, arm, start, inst, root))
    return jobs


def hms(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--src", required=True, help="frozen copy of the repo (freeze.sh)")
    ap.add_argument("--exp", nargs="+", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--arms", default="*", help="glob over arm names")
    ap.add_argument("--starts", default="*", help="glob over start labels")
    ap.add_argument("--instances", default="*", help="glob over instance names")
    ap.add_argument("--slots", type=int, default=7)
    ap.add_argument("--cores", default="0-6", help="cores to pin jobs to, e.g. 0-6 or 0,1,2,3")
    ap.add_argument("--mem-gb", type=float, default=56.0, help="budget for the sum of running jobs' estimates")
    ap.add_argument("--cgroup", action="store_true", help="per-job MemoryMax via systemd-run --user --scope")
    ap.add_argument("--skip-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--py", default=os.environ.get("PY", sys.executable))
    a = ap.parse_args()

    root, src = os.path.abspath(a.root), os.path.abspath(a.src)
    cores = []
    for part in a.cores.split(","):
        lo, _, hi = part.partition("-")
        cores += list(range(int(lo), int(hi or lo) + 1))
    slots = min(a.slots, len(cores))
    jobs = build_jobs(root, a.exp, a.arms, a.starts, a.instances)
    todo = [j for j in jobs if j.status() != "ok" and not (a.skip_failed and j.status() == "failed")]
    todo.sort(key=lambda j: -j.est)
    too_big = [j for j in todo if j.mem > a.mem_gb]
    total = sum(j.est for j in todo)
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run: ~{total / 3600:.0f} slot-h, "
          f"~{hms(total / slots)} wall at {slots} slots (estimates; memory budget {a.mem_gb:.0f} GB)")
    for e in a.exp:
        sub = [j for j in todo if j.exp.name == e]
        print(f"  {e:14s} {len(sub):4d} jobs  {sum(j.est for j in sub) / 3600:6.1f} slot-h  "
              f"max mem {max((j.mem for j in sub), default=0):5.1f} GB")
    if too_big:
        print(f"WARNING: {len(too_big)} jobs need more than --mem-gb and will never start, e.g. {too_big[0].id}")
    if a.dry_run:
        for j in todo[:15]:
            print(f"    {j.exp.name}/{j.id}  est {hms(j.est)}  mem {j.mem:.0f} GB")
            print("      " + " ".join(j.commands(a.py, src)[0]))
        return

    if not os.path.exists(os.path.join(src, "matheuristic.py")):
        sys.exit(f"no frozen code at {src} (run scripts/campaign/freeze.sh)")
    for j in todo:
        j.prepare(root)
    have_taskset = shutil.which("taskset") is not None
    gnu_time = next((t for t in ("/usr/bin/time", shutil.which("time"))
                     if t and os.path.exists(t) and subprocess.run([t, "-v", "true"], capture_output=True).returncode == 0),
                    None)
    if not have_taskset:
        print("WARNING: taskset not found: jobs are NOT pinned to cores (NixOS: util-linux)")
    if gnu_time is None:
        print("WARNING: GNU time not found: no peak-memory measurement (NixOS: nix-shell -p time, or add `time`)")
    progress = open(os.path.join(root, "progress.txt"), "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%m-%d %H:%M')} {msg}"
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    running: dict[int, tuple[Job, subprocess.Popen, int, float, list]] = {}  # pid -> (job, proc, core, t0, cmds)
    free_cores = cores[:slots]
    used_mem = 0.0
    n_done, t_start, done_est, done_wall = 0, time.time(), 0.0, 0.0
    log(f"start: {len(todo)} jobs of {','.join(a.exp)}, slots {slots}, cores {free_cores}, mem {a.mem_gb} GB, "
        f"src {src}")

    def launch(j: Job, core: int) -> subprocess.Popen:
        os.makedirs(os.path.join(j.exp_dir, "logs"), exist_ok=True)
        cmds = j.commands(a.py, src)
        prefix = []
        if a.cgroup:
            prefix += ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={int(1.5 * j.mem * 1024)}M"]
        if have_taskset:
            prefix += ["taskset", "-c", str(core)]
        if gnu_time:
            prefix += [gnu_time, "-v"]
        script = " && ".join(" ".join(_q(x) for x in prefix + c) for c in cmds)
        out = open(os.path.join(j.exp_dir, "logs", j.id + ".log"), "w")
        err = open(os.path.join(j.exp_dir, "logs", j.id + ".err"), "w")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MPLBACKEND="Agg", MPLCONFIGDIR=os.path.join(root, ".mpl"))
        p = subprocess.Popen(["bash", "-c", script], cwd=j.cwd, stdout=out, stderr=err, env=env,
                             start_new_session=True)
        p._cmds = cmds  # type: ignore[attr-defined]
        return p

    def stop_all(signum, frame):  # noqa: ARG001
        log(f"signal {signum}: stopping {len(running)} running jobs (they will be re-run)")
        for _, p, *_ in running.values():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, stop_all)
    signal.signal(signal.SIGINT, stop_all)

    queue = list(todo)
    while queue or running:
        started = True
        while started and free_cores:
            started = False
            for j in queue:
                if used_mem + j.mem <= a.mem_gb:
                    core = free_cores.pop(0)
                    p = launch(j, core)
                    running[p.pid] = (j, p, core, time.time(), p._cmds)
                    used_mem += j.mem
                    queue.remove(j)
                    started = True
                    break
        if not running and queue:
            sys.exit(f"cannot start {queue[0].id}: needs {queue[0].mem} GB > --mem-gb {a.mem_gb}")
        time.sleep(5)
        for pid, (j, p, core, t0, cmds) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[pid]
            free_cores.append(core)
            used_mem -= j.mem
            wall = time.time() - t0
            rss = _peak_rss_gb(os.path.join(j.exp_dir, "logs", j.id + ".err"))
            os.makedirs(os.path.dirname(j.done_file), exist_ok=True)
            with open(j.done_file, "w") as f:
                json.dump({"rc": rc, "wall_s": wall, "est_s": j.est, "peak_rss_gb": rss, "mem_est_gb": j.mem,
                           "core": core, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "commands": cmds}, f)
            n_done += 1
            done_est += j.est
            done_wall += wall
            # ETA: remaining estimated work, corrected by the measured wall/estimate ratio, over the slots
            ratio = done_wall / done_est if done_est else 1.0
            rem = (sum(x.est for x in queue) + sum(max(x.est - (time.time() - t), 0) for x, _, _, t, _ in
                                                   running.values())) * ratio
            mem_note = f" peak {rss:.1f}/{j.mem:.0f} GB" if rss is not None else ""
            log(f"[{n_done}/{len(todo)}] {'ok ' if rc == 0 else f'FAILED rc={rc}'} {j.exp.name}/{j.id} "
                f"{hms(wall)}{mem_note} | elapsed {hms(time.time() - t_start)}, ETA {hms(rem / slots)}")
            if rss is not None and rss > 1.2 * j.mem:
                log(f"    memory estimate too low for {j.exp.name}/{j.id}: update experiments.MEM_GB")
    log("finished")


def _q(x: str) -> str:
    return x if all(c.isalnum() or c in "-_./=:," for c in x) else "'" + x.replace("'", "'\\''") + "'"


def _peak_rss_gb(err_path: str) -> float | None:
    """Largest 'Maximum resident set size' (GNU time -v, kB) in the job's stderr."""
    best = None
    try:
        with open(err_path, errors="replace") as f:
            for line in f:
                if "Maximum resident set size" in line:
                    kb = int(line.rsplit(":", 1)[1])
                    best = max(best or 0.0, kb / 1024 ** 2)
    except (OSError, ValueError):
        pass
    return best


if __name__ == "__main__":
    main()
