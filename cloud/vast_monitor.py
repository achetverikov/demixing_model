#!/usr/bin/env python3
"""cloud/vast_monitor.py — Vast.ai pipeline monitor and auto-scaler.

Polls every --poll-interval seconds:
  • Kills instances that are: stuck in loading (>--loading-timeout min),
    crashed (exited/error status), or slow/errored (checked via logs after
    --slowness-check min of runtime).
  • Reports per-machine surface/hr from recent S3 bundle manifests
    and overall pipeline progress from Redis.
  • Scans the Vast.ai market and spawns new workers on offers whose
    estimated total pipeline cost is below --budget USD.

Usage:
  cloud/vast_monitor.py [--budget 30] [--max-concurrent 10] [--dry-run]
  cloud/vast_monitor.py --registry averaged_surfaces_vast \\
      --s3-prefix demixing/surfaces_full --n-samples 20
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

# ── Pipeline constants ──────────────────────────────────────────────────────
N_SURFACES   = 32_800
STORAGE_RATE = 0.0306   # $/hr median for 60 GB disk (from invoice)

# Expected surfaces/hr per GPU type (NV image, n_samples=20, outliers removed)
EXPECTED_SPH: dict[str, float] = {
    'RTX_3090':         756,
    'RTX_4090':        1141,
    'RTX_5060_Ti':      498,
    'RTX_5090':        1292,
    'RTX_PRO_6000_WS': 1426,
    'H100_NVL':        1456,
    'A100_SXM4':        884,
}

# Log patterns that signal a fatal instance error
_FATAL_PATTERNS = [
    r'no supported devices found',
    r'docker login failed',
    r'Error: GPU error',
    r'CUDA_ERROR_NO_DEVICE',
    r'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    r'Failed to pull image',
    r'Jax is running on device \[CpuDevice',   # GPU init failed, fell back to CPU
    r'Falling back to cpu',
]
_FATAL_RE = re.compile('|'.join(_FATAL_PATTERNS), re.IGNORECASE)

# SSH/network noise that should be ignored in log health checks
_SSH_NOISE_RE = re.compile(
    r'ssh.*connection.*refused|'
    r'ssh.*timed out|'
    r'connection closed|'
    r'kex_exchange_identification|'
    r'unable to connect',
    re.IGNORECASE,
)

# Log patterns that signal active surface computation
_ACTIVE_PATTERNS = re.compile(
    r'Saved averaged_|Bundled \d+ surfaces|Pipeline: sf|Processing L\d_chunk',
    re.IGNORECASE,
)


# ── GPU name normalisation ──────────────────────────────────────────────────
def norm_gpu(name: str) -> str:
    """'RTX 4090' → 'RTX_4090'"""
    return re.sub(r'[\s/]+', '_', name.strip())


# ── Environment loading ─────────────────────────────────────────────────────
def load_env_file(path: str) -> None:
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                line = line.removeprefix('export ')
                k, _, v = line.partition('=')
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


# ── Vastai wrappers ─────────────────────────────────────────────────────────
def _vastai() -> str:
    b = shutil.which('vastai')
    if b:
        return b
    venv = Path(sys.executable).parent / 'vastai'
    if venv.exists():
        return str(venv)
    raise RuntimeError('vastai CLI not found; run: pip install vastai')


def _run_vastai(*args, timeout: int = 30) -> Optional[object]:
    """Run vastai and return parsed JSON, or None on error."""
    try:
        r = subprocess.run(
            [_vastai()] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            msg = r.stderr.strip() or r.stdout.strip()
            print(f"  [vastai] {msg[:140]}", file=sys.stderr)
            return None
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  [vastai error] {e}", file=sys.stderr)
        return None


def get_instances() -> list[dict]:
    data = _run_vastai('show', 'instances-v1', '--raw')
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # instances-v1 returns {"instances": [...], ...}
        return data.get('instances', list(data.values()))
    return []


def search_offers(gpu_name: str, disk_gb: int) -> list[dict]:
    """Search for on-demand offers for a given GPU (internal name, e.g. RTX_4090)."""
    gpu_search = gpu_name.replace('_', ' ')
    query = (f'gpu_name="{gpu_search}" num_gpus=1 '
             f'disk_space>={disk_gb} reliability>0.8')
    data = _run_vastai('search', 'offers', '--raw', '--type', 'on-demand', query)
    if isinstance(data, dict):
        return data.get('offers', [])
    if isinstance(data, list):
        return data
    return []


def fetch_logs(instance_id: int, n_lines: int = 80,
               log_dir: Optional[Path] = None,
               print_tail: bool = False) -> str:
    """Fetch container log tail; optionally save to file and/or print to terminal."""
    def _tail(cmd) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            lines = (r.stdout or '').splitlines()
            return '\n'.join(lines[-n_lines:])
        except Exception:
            return ''

    v = _vastai()
    container_log = _tail([v, 'logs', str(instance_id)])
    daemon_log    = _tail([v, 'logs', '--daemon-logs', str(instance_id)])

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        if container_log:
            (log_dir / f'{instance_id}.log').write_text(container_log)
        if daemon_log:
            (log_dir / f'{instance_id}_daemon.log').write_text(daemon_log)

    if print_tail:
        for label, text in [('container', container_log), ('daemon', daemon_log)]:
            if text:
                print(f"    ── {label} log (last {n_lines} lines) ──")
                for line in text.splitlines():
                    print(f"    {line}")
            else:
                print(f"    ── {label} log: (empty) ──")

    return container_log


def destroy_instance(instance_id: int, reason: str, dry_run: bool) -> None:
    tag = '[DRY RUN] ' if dry_run else ''
    print(f"  {tag}⚠  Destroying {instance_id}: {reason}")
    if not dry_run:
        subprocess.run(
            ['bash', '-c', f'echo y | {_vastai()} destroy instance {instance_id}'],
            capture_output=True, timeout=30,
        )


def spawn_worker(offer_id: int, env_dict: dict, onstart: str, image: str,
                 disk: int, docker_user: str, docker_pass: str,
                 dry_run: bool) -> Optional[int]:
    env_str = ' '.join(f'-e {k}={v}' for k, v in env_dict.items())
    cmd = [
        _vastai(), 'create', 'instance', str(offer_id),
        '--image', image,
        '--disk', str(disk),
        '--ssh', '--direct',
        '--env', env_str,
        '--onstart-cmd', onstart,
    ]
    if docker_user and docker_pass:
        cmd += ['--login', f'-u {docker_user} -p {docker_pass} docker.io']

    tag = '[DRY RUN] ' if dry_run else ''
    gpu_hint = env_dict.get('VAST_GPU_HINT', offer_id)
    print(f"  {tag}+  Spawning on offer {offer_id} ({gpu_hint})")
    if dry_run:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        match = re.search(r"new_contract['\": ]+(\d+)", r.stdout)
        if match:
            return int(match.group(1))
        print(f"  [spawn warn] {r.stdout[:200]}")
        return None
    except Exception as e:
        print(f"  [spawn error] {e}")
        return None


# ── S3 / progress helpers ───────────────────────────────────────────────────
def _make_s3():
    import boto3
    return boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'auto'),
    )


def get_recent_manifests(s3, bucket: str, prefix: str,
                         window_min: float) -> list[dict]:
    """Fetch manifests modified within the last window_min minutes."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
    pfx = f"{prefix}/surface_bundle_" if prefix else "surface_bundle_"
    pager = s3.get_paginator('list_objects_v2')
    results = []
    for page in pager.paginate(Bucket=bucket, Prefix=pfx):
        for obj in page.get('Contents', []):
            if not obj['Key'].endswith('.manifest.json'):
                continue
            if obj['LastModified'] < cutoff:
                continue
            try:
                body = s3.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read()
                m = json.loads(body)
                m['_ts'] = obj['LastModified']
                results.append(m)
            except Exception:
                pass
    return results


def count_completed_s3(s3, bucket: str, prefix: str) -> int:
    """Count surfaces from all manifests (used only if Redis unavailable)."""
    pfx = f"{prefix}/surface_bundle_" if prefix else "surface_bundle_"
    pager = s3.get_paginator('list_objects_v2')
    total = 0
    for page in pager.paginate(Bucket=bucket, Prefix=pfx):
        total += sum(1 for o in page.get('Contents', [])
                     if o['Key'].endswith('.manifest.json'))
    return total * 5  # approximate: default chunk_size=5


def redis_completed(registry: str) -> int:
    from lock_backend import make_lock_backend
    backend = make_lock_backend('redis')
    return len(backend.completed_members(registry))


# ── Actual spend from Vast billing API ─────────────────────────────────────
def fetch_billed_spend(our_instance_ids: set) -> dict[int, float]:
    """Return {instance_id: billed_amount} for our instances from the charges API."""
    billed: dict[int, float] = {}
    next_token = None
    while True:
        cmd = [_vastai(), 'show', 'invoices-v1', '--charges', '--raw', '--full',
               '--limit', '100']
        if next_token:
            cmd += ['--next-token', next_token]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(r.stdout)
        except Exception:
            break
        for entry in data.get('results', []):
            src = entry.get('source', '')
            if not src.startswith('instance-'):
                continue
            try:
                iid = int(src.removeprefix('instance-'))
            except ValueError:
                continue
            if iid in our_instance_ids:
                billed[iid] = billed.get(iid, 0.0) + float(entry.get('amount', 0))
        next_token = data.get('next_token')
        if not next_token:
            break
    return billed


# ── Budget ──────────────────────────────────────────────────────────────────
def pipeline_cost(gpu: str, price_per_hour: float) -> Optional[float]:
    sph = EXPECTED_SPH.get(gpu)
    if not sph:
        return None
    return (price_per_hour + STORAGE_RATE) * (N_SURFACES / sph)


# ── State persistence ───────────────────────────────────────────────────────
STATE_FILE = ROOT / 'logs' / 'vast_monitor_state.json'


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {'spawned': {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ── Display helpers ─────────────────────────────────────────────────────────
def _bar(n: int, total: int = N_SURFACES, width: int = 28) -> str:
    filled = int(width * n / max(total, 1))
    pct = 100 * n / max(total, 1)
    return f"[{'█'*filled}{'░'*(width-filled)}] {n:>6,}/{total:,}  {pct:5.1f}%"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ── Main loop ───────────────────────────────────────────────────────────────
def run_loop(args: argparse.Namespace) -> None:
    state = load_state()
    s3 = _make_s3()
    bucket = os.environ['S3_BUCKET']
    poll = 0

    while True:
        poll += 1
        now = datetime.now(timezone.utc)
        print(f"\n{'═'*72}")
        print(f"  Vast Monitor  poll #{poll}  "
              f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'═'*72}")

        # ── Fetch instances ──────────────────────────────────────────────
        instances = {inst['id']: inst
                     for inst in get_instances()
                     if isinstance(inst, dict) and 'id' in inst}

        # Drop state for instances that have been removed by vastai; add tombstone
        tombstones = state.setdefault('tombstones', {})
        for iid in list(state['spawned']):
            if int(iid) not in instances:
                meta = state['spawned'][iid]
                if meta.get('kill_reason'):
                    msg = (f"  ✓  {iid} ({meta.get('gpu_name','?')}) destroyed — "
                           f"reason: {meta['kill_reason']}")
                else:
                    last = meta.get('last_status', 'unknown')
                    age  = meta.get('last_age_min', '?')
                    msg  = (f"  ⚠  {iid} ({meta.get('gpu_name','?')}) vanished — "
                            f"last seen: {last} at {age} min "
                            f"(host preemption or mid-run crash)")
                tombstones[iid] = {'msg': msg, 'remaining': 10}
                del state['spawned'][iid]

        # Print and age tombstones
        for iid in list(tombstones):
            print(tombstones[iid]['msg'])
            tombstones[iid]['remaining'] -= 1
            if tombstones[iid]['remaining'] <= 0:
                del tombstones[iid]

        # ── Recent S3 throughput ─────────────────────────────────────────
        try:
            recent = get_recent_manifests(
                s3, bucket, args.s3_prefix, args.throughput_window)
        except Exception as e:
            print(f"  [S3 warn] {e}")
            recent = []

        # Group by the machine_id stored in each manifest
        machine_sfc: dict[str, int] = defaultdict(int)
        for m in recent:
            mid = m.get('machine_id', 'unknown')
            machine_sfc[mid] += m.get('surface_count', 0)
        # Convert counts in window → surfaces/hr
        machine_sph: dict[str, float] = {
            mid: cnt / (args.throughput_window / 60.0)
            for mid, cnt in machine_sfc.items()
        }
        total_recent_sph = sum(machine_sph.values())

        # ── Overall progress ─────────────────────────────────────────────
        progress_count = 0
        progress_source = '?'
        if os.environ.get('REDIS_HOST') and args.registry:
            try:
                progress_count = redis_completed(args.registry)
                progress_source = 'redis'
            except Exception as e:
                print(f"  [Redis] {e}")
        if progress_count == 0:
            try:
                progress_count = count_completed_s3(s3, bucket, args.s3_prefix)
                progress_source = 'S3 ≈'
            except Exception:
                pass

        print(f"\n  Progress ({progress_source}): {_bar(progress_count)}")
        if total_recent_sph > 0:
            remaining = max(0, N_SURFACES - progress_count)
            eta_h = remaining / total_recent_sph
            print(f"  Throughput : {total_recent_sph:.0f} sph  •  ETA ≈ {eta_h:.1f} h")
        else:
            print(f"  Throughput : no surface activity in last "
                  f"{args.throughput_window:.0f} min")

        do_log_check = (poll % args.log_check_every == 0)

        # ── Actual spend (fetched every log_check_every polls) ───────────
        if do_log_check:
            our_ids = {int(iid) for iid in state['spawned']}
            # also include tombstoned instances from this session
            for iid in tombstones:
                try:
                    our_ids.add(int(iid))
                except ValueError:
                    pass
            try:
                billed = fetch_billed_spend(our_ids)
                state['_billed'] = billed
            except Exception as e:
                print(f"  [billing warn] {e}")
        billed = {int(k): v for k, v in state.get('_billed', {}).items()}

        # Running estimate: dph * age for instances still live
        running_est = 0.0
        for inst in instances.values():
            if str(inst['id']) not in state['spawned']:
                continue
            start_ts = inst.get('start_date') or inst.get('start_time') or 0
            if start_ts:
                age_h = (now.timestamp() - float(start_ts)) / 3600.0
                running_est += float(inst.get('dph_total') or 0) * age_h

        total_billed  = sum(billed.values())
        total_spend   = total_billed + running_est
        budget_pct    = 100 * total_spend / args.budget
        print(f"  Budget     : ${total_billed:.2f} billed  +  ~${running_est:.2f} running"
              f"  =  ~${total_spend:.2f} / ${args.budget:.0f}"
              f"  ({budget_pct:.0f}%)")

        # ── Instance health ──────────────────────────────────────────────
        # machine_sph keys are "<instance_id>-gpu0" for new workers (CONTAINER_ID-based)
        # or a container hostname prefix for old workers — normalise to instance ID where possible
        def _sph_for(iid: int) -> Optional[float]:
            key = f"{iid}-gpu0"
            if key in machine_sph:
                return machine_sph[key]
            # Multi-GPU: sum all gpu slots for this instance
            total = sum(v for k, v in machine_sph.items() if k.startswith(f"{iid}-gpu"))
            return total if total else None

        print(f"\n  Instances ({len(instances)})  "
              f"[throughput = last {args.throughput_window:.0f} min]:")
        active_count = 0

        for iid, inst in sorted(instances.items()):
            # Vast raw JSON uses actual_status / cur_state, not 'status'
            actual_st   = inst.get('actual_status', '')
            cur_state   = inst.get('cur_state', '')
            status      = actual_st or cur_state
            gpu_raw     = inst.get('gpu_name', '')
            gpu         = norm_gpu(gpu_raw)
            dph         = float(inst.get('dph_total') or 0)
            start_ts    = inst.get('start_date') or inst.get('start_time') or 0
            is_ours     = str(iid) in state['spawned']

            age_min = 0.0
            if start_ts:
                started = datetime.fromtimestamp(float(start_ts), tz=timezone.utc)
                age_min = (now - started).total_seconds() / 60.0

            cost_est  = pipeline_cost(gpu, dph)
            cost_str  = f'~${cost_est:5.1f}' if cost_est else '    ?  '
            flag      = '[ours]' if is_ours else '[ext] '
            sph       = _sph_for(iid)
            sph_str   = f'{sph:>5.0f} sph' if sph else '    --   '
            print(f"    {flag} {iid:<12} {gpu:<22} {status:<12} "
                  f"{age_min:5.0f}min  ${dph:.4f}/hr  {cost_str}  {sph_str}")

            kill_reason: Optional[str] = None

            # Snapshot prev_status before updating so we can detect transitions
            prev_status = (state['spawned'].get(str(iid), {}).get('last_status', '')
                           if is_ours else '')

            # Update last-known status so vanish messages are informative
            if is_ours:
                state['spawned'][str(iid)]['last_status'] = status
                state['spawned'][str(iid)]['last_age_min'] = round(age_min)

            # Immediately fetch + print logs when instance first enters running state
            if (is_ours and status == 'running'
                    and prev_status not in ('running', '')):
                print(f"    → {iid} transitioned {prev_status!r} → running — fetching logs")
                fetch_logs(iid, log_dir=ROOT / 'logs' / 'instance_logs',
                           print_tail=True)

            if is_ours or args.manage_all:
                # Stuck loading / being created
                if status in ('loading', 'created', 'provisioning'):
                    if age_min > args.loading_timeout:
                        kill_reason = (f'stuck in {status!r} for '
                                       f'{age_min:.0f} min')
                    elif actual_st == 'loading' and cur_state == 'stopped':
                        kill_reason = 'container stopped while loading (failed to start)'

                # Crashed
                elif status in ('exited', 'stopped', 'error', 'failed', 'dead'):
                    kill_reason = f'crashed (actual_status={actual_st!r})'

                # Running — check logs every log_check_every polls
                elif status == 'running' and do_log_check:
                    logs = fetch_logs(iid, log_dir=ROOT / 'logs' / 'instance_logs')
                    # Ignore polls where vastai logs only returns SSH noise
                    if _SSH_NOISE_RE.search(logs) and not _ACTIVE_PATTERNS.search(logs):
                        pass  # container not yet reachable via SSH, skip check
                    elif _FATAL_RE.search(logs):
                        kill_reason = 'fatal error in logs'
                    elif age_min > args.slowness_check:
                        # Use both SSH log AND S3 manifests as activity signals.
                        # Kill only if neither source shows work; warn if they disagree
                        # (log-active but no S3 → upload problem; S3-active but quiet
                        # log → worker between chunks, not a kill condition).
                        log_active = bool(_ACTIVE_PATTERNS.search(logs))
                        s3_active  = (_sph_for(iid) or 0) > 0
                        if not log_active and not s3_active:
                            kill_reason = (f'no activity in log or S3 after '
                                           f'{age_min:.0f} min')
                        elif log_active and not s3_active:
                            print(f"    ⚠  {iid}: log active but no S3 manifests "
                                  f"in last {args.throughput_window:.0f} min "
                                  f"(possible S3/Redis sync issue)")
                        elif not log_active and s3_active:
                            sph_val = _sph_for(iid)
                            print(f"    ℹ  {iid}: S3={sph_val:.0f} sph but quiet log "
                                  f"(likely between chunks)")

            if kill_reason:
                fetch_logs(iid, log_dir=ROOT / 'logs' / 'instance_logs',
                           print_tail=True)
                destroy_instance(iid, kill_reason, args.dry_run)
                # Keep in state so next poll can report the confirmed kill
                state['spawned'][str(iid)]['kill_reason'] = kill_reason
                state['spawned'][str(iid)]['killed_at'] = _now_iso()
            elif status not in ('exited', 'stopped', 'error', 'failed', 'dead'):
                active_count += 1

        # Show throughput for unmatched machine_ids (pre-fix workers using hostnames)
        matched_keys = {f"{iid}-gpu{g}" for iid in instances for g in range(8)}
        unmatched = {mid: sph for mid, sph in machine_sph.items()
                     if mid not in matched_keys}
        if unmatched:
            print(f"    {'(unmatched legacy workers)':<36}", end='')
            for mid, sph in sorted(unmatched.items(), key=lambda x: -x[1]):
                print(f"  {mid}: {sph:.0f} sph", end='')
            print()

        # ── Market scan and spawning ─────────────────────────────────────
        slots = max(0, args.max_concurrent - active_count)
        print(f"\n  Running: {active_count}  "
              f"Target: {args.max_concurrent}  Slots open: {slots}")

        if progress_count >= N_SURFACES:
            print(f"\n  ✓ Pipeline complete — all {N_SURFACES:,} surfaces done.")
        elif slots > 0:
            print(f"\n  Scanning market (budget ${args.budget:.0f}, "
                  f"{slots} slot{'s' if slots != 1 else ''})…")
            spawned = 0
            used_hosts = {inst.get('host_id') for inst in instances.values()}

            for gpu, _ in sorted(EXPECTED_SPH.items(),
                                  key=lambda kv: -kv[1]):   # fastest first
                if spawned >= slots:
                    break
                offers = search_offers(gpu, args.disk)
                offers.sort(key=lambda o: o.get('dph_total', 999))

                for offer in offers:
                    if spawned >= slots:
                        break
                    oid      = offer.get('id') or offer.get('ask_contract_id')
                    price    = float(offer.get('dph_total') or 0)
                    geo      = offer.get('geolocation', '')
                    host_id  = offer.get('host_id')
                    disk_avl = offer.get('disk_space', 0)

                    # Exclusions
                    if 'China' in geo or ', CN' in geo:
                        continue
                    if host_id in used_hosts:
                        continue
                    if disk_avl < args.disk:
                        continue

                    cost = pipeline_cost(gpu, price)
                    if cost is None or cost > args.budget:
                        break   # sorted by price; cheaper offers exhausted

                    print(f"    ✓ {gpu:<22} offer {oid}  "
                          f"${price:.4f}/hr  ~${cost:.1f} pipeline  {geo}")

                    registry  = args.registry or 'averaged_surfaces_vast'
                    avg_dir   = f'results/{registry}'
                    bundle_dir = f'results/{registry}_bundles'
                    env = {
                        'GIT_REPO':            os.environ.get('GIT_REPO',
                            'https://github.com/achetverikov/demixing_model.git'),
                        'GIT_REF':             os.environ.get('GIT_REF', 'main'),
                        'REPO_DIR':            '/workspace/demixing_model',
                        'REDIS_HOST':          os.environ['REDIS_HOST'],
                        'REDIS_PORT':          os.environ.get('REDIS_PORT', '6379'),
                        'REDIS_USERNAME':      os.environ.get('REDIS_USERNAME', 'default'),
                        'REDIS_PASSWORD':      os.environ['REDIS_PASSWORD'],
                        'S3_BUCKET':           os.environ['S3_BUCKET'],
                        'S3_PREFIX':           args.s3_prefix,
                        'S3_ENDPOINT_URL':     os.environ['S3_ENDPOINT_URL'],
                        'AWS_ACCESS_KEY_ID':   os.environ['AWS_ACCESS_KEY_ID'],
                        'AWS_SECRET_ACCESS_KEY': os.environ['AWS_SECRET_ACCESS_KEY'],
                        'AWS_DEFAULT_REGION':  os.environ.get('AWS_DEFAULT_REGION', 'auto'),
                        'AVERAGED_SURFACES_DIR': avg_dir,
                        'COMPLETION_REGISTRY': registry,
                        'BUNDLE_OUTPUT':       '1',
                        'BUNDLE_DIR':          bundle_dir,
                        'N_SIMULATIONS':       str(args.n_simulations),
                        'N_SAMPLES':           str(args.n_samples),
                        'VAST_GPU_HINT':        gpu,
                    }
                    onstart = (
                        'cd /workspace'
                        ' && git clone "$GIT_REPO" "$REPO_DIR"'
                        ' && cd "$REPO_DIR"'
                        ' && git checkout "$GIT_REF"'
                        ' && bash cloud/vast_worker.sh;'
                        ' echo y | vastai destroy instance $CONTAINER_ID'
                    )
                    iid = spawn_worker(
                        oid, env, onstart, args.image, args.disk,
                        os.environ.get('DOCKER_LOGIN_USER', ''),
                        os.environ.get('DOCKER_LOGIN_PASS', ''),
                        args.dry_run,
                    )
                    if iid or args.dry_run:
                        spawned += 1
                        used_hosts.add(host_id)
                        if iid:
                            state['spawned'][str(iid)] = {
                                'spawned_at': _now_iso(),
                                'offer_id':   oid,
                                'gpu_name':   gpu,
                                'price_per_hour': price,
                                'estimated_cost': cost,
                            }
                        break   # one instance per GPU type per round
            if spawned == 0:
                print(f"    No suitable offers found within ${args.budget:.0f}")

        save_state(state)
        print(f"\n  Next poll in {args.poll_interval}s  "
              f"(log check every {args.log_check_every} polls)  Ctrl+C to stop")
        try:
            time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            print("\n  Monitor stopped.")
            break


# ── Entry point ─────────────────────────────────────────────────────────────
def main() -> int:
    load_env_file(str(ROOT / '.env.docker'))
    load_env_file(str(ROOT / '.env'))

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--budget', type=float, default=30.0,
                    help='Max estimated pipeline cost in USD (default: 30)')
    ap.add_argument('--max-concurrent', type=int, default=10,
                    help='Max simultaneous running instances (default: 10)')
    ap.add_argument('--poll-interval', type=int, default=60,
                    help='Seconds between polls (default: 60)')
    ap.add_argument('--registry',
                    default=os.environ.get('COMPLETION_REGISTRY', ''),
                    help='Redis completion registry (default: $COMPLETION_REGISTRY)')
    ap.add_argument('--s3-prefix',
                    default=os.environ.get('S3_PREFIX', 'demixing/surfaces_full'),
                    help='S3 prefix for surface bundles')
    ap.add_argument('--n-simulations', type=int, default=10000,
                    help='Simulations per surface (default: 10000)')
    ap.add_argument('--n-samples', type=int, default=20,
                    help='Samples per surface (default: 20)')
    ap.add_argument('--image',
                    default=os.environ.get('VAST_IMAGE',
                                           'andreychetverikov/demixing-vast:latest'),
                    help='Docker image for workers')
    ap.add_argument('--disk', type=int, default=60,
                    help='Disk GB per instance (default: 60)')
    ap.add_argument('--loading-timeout', type=float, default=10.0,
                    help='Minutes before killing a stuck-loading instance (default: 10)')
    ap.add_argument('--slowness-check', type=float, default=30.0,
                    help='Minutes before log-checking a running instance (default: 30)')
    ap.add_argument('--throughput-window', type=float, default=10.0,
                    help='Minutes window for surface/hr calculation (default: 10)')
    ap.add_argument('--log-check-every', type=int, default=3,
                    help='Check instance logs every N polls (default: 3)')
    ap.add_argument('--manage-all', action='store_true',
                    help='Apply kill logic to all instances, not just ones spawned by this monitor')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report actions without executing them')
    args = ap.parse_args()

    # Activate vastai API key from env
    key = os.environ.get('VASTAPIKEY') or os.environ.get('VAST_API_KEY', '')
    if key:
        subprocess.run([_vastai(), 'set', 'api-key', key], capture_output=True)

    try:
        run_loop(args)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
