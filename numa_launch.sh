#!/usr/bin/env bash
# NUMA-aware torchrun launcher for SlideFSDP (improved).
# Usage:
#   torchrun ... --no_python ./numa_launch.sh -- train.py ...

set -euo pipefail

# -----------------------
# Configuration
# -----------------------
CONTROLLER_CORES=${CONTROLLER_CORES:-1}   # physical cores per rank for "control" (per NUMA node)
PYTHON_BIN=${PYTHON_BIN:-python}

# USE_FULL_NODE_FOR_RANK0:
# 0: Local NUMA Mode  -> rank0 takes spare cores only from its NUMA node
# 1: Global Node Mode -> rank0 takes spare cores from ALL NUMA nodes (whole host)
USE_FULL_NODE_FOR_RANK0=${USE_FULL_NODE_FOR_RANK0:-0}

export CONTROLLER_CORES
export USE_FULL_NODE_FOR_RANK0
export LOCAL_RANK=${LOCAL_RANK:-0}
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-${WORLD_SIZE:-1}}
export NUMA_LAUNCH_DEBUG=${NUMA_LAUNCH_DEBUG:-0}

if [[ "$#" -eq 0 ]]; then
  echo "Usage: torchrun ... --no_python $0 -- <python_entrypoint> [args...]" >&2
  exit 1
fi

topology_script=$(cat <<'EOF'
import os
import sys
import glob
import subprocess
from collections import defaultdict

def debug(msg):
    if os.environ.get("NUMA_LAUNCH_DEBUG", "0") == "1":
        sys.stderr.write(f"[numa_top] {msg}\n")

def get_visible_cpus():
    # Respect docker/cgroup cpuset
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        # Fallback: assume all online CPUs
        cpu_dirs = glob.glob("/sys/devices/system/cpu/cpu[0-9]*")
        cpus = []
        for d in cpu_dirs:
            base = os.path.basename(d)
            if base.startswith("cpu") and base[3:].isdigit():
                cpus.append(int(base[3:]))
        return sorted(cpus)

def get_cpu_siblings(visible_set):
    """
    Returns:
      siblings: { (pkg_id, core_id): [cpu_id, sibling_cpu_id, ...] }
      cpu_to_node: { cpu_id: numa_node }
    Only includes CPUs visible in current cpuset.
    """
    siblings = {}
    cpu_to_node = {}

    cpu_dirs = glob.glob("/sys/devices/system/cpu/cpu[0-9]*")
    for cpu_dir in cpu_dirs:
        try:
            cpu_name = os.path.basename(cpu_dir)
            if not cpu_name[3:].isdigit():
                continue
            cpu_id = int(cpu_name.replace("cpu", ""))

            if cpu_id not in visible_set:
                continue

            # Filter offline CPUs (if exposed)
            online_path = os.path.join(cpu_dir, "online")
            if os.path.exists(online_path):
                try:
                    if open(online_path).read().strip() == "0":
                        continue
                except Exception:
                    pass

            cid_path = os.path.join(cpu_dir, "topology/core_id")
            if not os.path.exists(cid_path):
                continue
            core_id = int(open(cid_path).read().strip())

            pid_path = os.path.join(cpu_dir, "topology/physical_package_id")
            pkg_id = 0
            if os.path.exists(pid_path):
                pkg_id = int(open(pid_path).read().strip())

            key = (pkg_id, core_id)
            siblings.setdefault(key, []).append(cpu_id)

            # NUMA node
            node_id = 0
            node_dirs = glob.glob(os.path.join(cpu_dir, "node*"))
            for nd in node_dirs:
                base = os.path.basename(nd)
                if base.startswith("node") and base[4:].isdigit():
                    node_id = int(base[4:])
                    break
            cpu_to_node[cpu_id] = node_id

        except Exception:
            continue

    # sort sibling lists
    for k in list(siblings.keys()):
        siblings[k] = sorted(siblings[k])

    return siblings, cpu_to_node

def get_gpu_numa_nodes():
    """
    Returns list of NUMA nodes for GPUs (respect CUDA_VISIBLE_DEVICES).
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        bus_ids = [x.strip() for x in out.splitlines() if x.strip()]
    except Exception:
        debug("nvidia-smi failed, fallback to heuristic")
        return []

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        idxs = [int(x) for x in cvd.split(",") if x.strip()]
        bus_ids = [bus_ids[i] for i in idxs if i < len(bus_ids)]

    nodes = []
    for bus in bus_ids:
        matches = glob.glob(f"/sys/bus/pci/devices/*{bus.lower()}/numa_node")
        node = -1
        if matches:
            try:
                node = int(open(matches[0]).read().strip())
            except Exception:
                pass
        if node < 0:
            node = 0
        nodes.append(node)

    return nodes

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    controller_cores_req = int(os.environ.get("CONTROLLER_CORES", "1"))
    use_full_node_rank0 = int(os.environ.get("USE_FULL_NODE_FOR_RANK0", "0"))

    visible_cpus = get_visible_cpus()
    visible_set = set(visible_cpus)

    # 1) CPU topology (only visible CPUs)
    siblings_map, cpu_to_node = get_cpu_siblings(visible_set)

    # node_cores_map: nid -> list of "physical cores", each item is list of sibling cpu_ids
    node_cores_map = defaultdict(list)
    for _, cpus in siblings_map.items():
        if not cpus:
            continue
        nid = cpu_to_node.get(cpus[0], 0)
        node_cores_map[nid].append(cpus)

    for nid in node_cores_map:
        node_cores_map[nid].sort(key=lambda x: x[0])

    # 2) rank -> node mapping (prefer GPU NUMA)
    gpu_nodes = get_gpu_numa_nodes()
    rank_to_node = {}

    if gpu_nodes:
        for r in range(local_world_size):
            if r < len(gpu_nodes):
                rank_to_node[r] = gpu_nodes[r]
            else:
                rank_to_node[r] = gpu_nodes[0]
    else:
        # fallback: round-robin across available NUMA nodes
        nodes = sorted(node_cores_map.keys()) if node_cores_map else [0]
        if not nodes:
            nodes = [0]
        for r in range(local_world_size):
            rank_to_node[r] = nodes[r % len(nodes)]

    node_to_ranks = defaultdict(list)
    for r, nid in rank_to_node.items():
        node_to_ranks[nid].append(r)
    for nid in node_to_ranks:
        node_to_ranks[nid].sort()

    # 3) Allocate control cores per rank on each NUMA node, then define spare(update) pool.
    rank_control_phys = {}           # r -> list of physical cores (each is list of sibling cpus)
    local_update_phys = defaultdict(list)  # nid -> spare physical cores
    global_update_phys = []          # all spare physical cores

    for nid in sorted(node_cores_map.keys()):
        phys = list(node_cores_map[nid])
        ranks = node_to_ranks.get(nid, [])
        needed = len(ranks) * controller_cores_req

        if len(phys) < needed:
            # shortage: evenly split whatever exists, no spare guarantees
            chunk = max(1, len(phys) // (len(ranks) if ranks else 1))
            for i, r in enumerate(ranks):
                s = i * chunk
                e = min(len(phys), s + chunk)
                rank_control_phys[r] = phys[s:e]
            # no spares
            continue

        # normal: carve control from beginning
        idx = 0
        for r in ranks:
            rank_control_phys[r] = phys[idx: idx + controller_cores_req]
            idx += controller_cores_req

        spares = phys[idx:]
        local_update_phys[nid] = spares
        global_update_phys.extend(spares)

    my_node = rank_to_node.get(local_rank, 0)
    my_control = rank_control_phys.get(local_rank, [])

    my_update = []
    if local_rank == 0:
        if use_full_node_rank0 == 1:
            my_update = global_update_phys
        else:
            my_update = local_update_phys.get(my_node, [])

    # flatten to absolute cpu ids
    def flatten_phys(phys_list):
        out = []
        for core in phys_list:
            out.extend(core)
        return out

    flat_control = flatten_phys(my_control)          # abs cpu ids
    flat_update  = flatten_phys(my_update)           # abs cpu ids

    # bind set (abs ids) for numactl
    if local_rank == 0:
        bind_abs = flat_control + flat_update
    else:
        bind_abs = flat_control

    # OMP threads: rank0 uses update threads; workers minimal
    if local_rank == 0:
        omp_threads = len(flat_update) if len(flat_update) > 0 else max(1, len(flat_control))
    else:
        # usually comm/control doesn't need big OMP; keep small
        omp_threads = max(1, len(flat_control))

    # For libgomp: pin OpenMP worker threads to update cores (ABS cpu ids) so they don't touch control cores.
    # If no update cores, do not set affinity.
    gomp_aff = ""
    if local_rank == 0 and len(flat_update) > 0:
        gomp_aff = ",".join(str(x) for x in flat_update)

    # exports
    print(f"export NUMA_LAUNCH_NODE={my_node}")
    print(f"export BIND_CPUS_STR={','.join(str(x) for x in bind_abs)}")
    print(f"export OMP_THREADS={omp_threads}")
    print(f"export NUMA_CPU_ADAM_CORES={','.join(str(x) for x in flat_update)}")
    print(f"export NUMA_CONTROL_CORES={','.join(str(x) for x in flat_control)}")

    # libgomp pinning (rank0 only)
    if gomp_aff:
        print(f"export GOMP_CPU_AFFINITY={gomp_aff}")
    else:
        # ensure it's not inherited from environment
        print("unset GOMP_CPU_AFFINITY")

    # Debug log
    log = f"[numa_launch] Rank={local_rank} Node={my_node} | "
    log += f"ControlThreads={len(flat_control)} [{','.join(str(x) for x in flat_control)}] | "
    if local_rank == 0:
        mode = "Global" if use_full_node_rank0 == 1 else "Local"
        log += f"UpdateThreads={len(flat_update)} Mode={mode} | "
        log += f"GOMP_CPU_AFFINITY={'set' if gomp_aff else 'unset'}"
    else:
        log += "Worker"
    sys.stderr.write(log + "\n")

if __name__ == "__main__":
    main()
EOF
)

# Execute Python script to get topology vars
eval "$($PYTHON_BIN -c "$topology_script")"

# -----------------------
# Runtime env (OpenMP/MKL)
# -----------------------
export OMP_NUM_THREADS="${OMP_THREADS}"
export MKL_NUM_THREADS="${OMP_THREADS}"

# Make OpenMP threads bind (actual CPU selection is controlled by GOMP_CPU_AFFINITY for rank0)
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-threads}

# Informational log
# echo "[numa_launch] Rank=${LOCAL_RANK} Node=${NUMA_LAUNCH_NODE} BindCPUs=${BIND_CPUS_STR} OMP_NUM_THREADS=${OMP_NUM_THREADS} GOMP_CPU_AFFINITY=${GOMP_CPU_AFFINITY-<unset>}" >&2

# -----------------------
# Execute with numactl
# Memory policy: preferred (local first, spillover allowed)
# -----------------------
exec numactl --physcpubind="${BIND_CPUS_STR}" --preferred="${NUMA_LAUNCH_NODE}" \
  ${PYTHON_BIN} "$@"
