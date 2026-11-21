#!/usr/bin/env python3
"""Create at most one recovery VM from one Cloud Shell session. Python stdlib only.

Preview by default. --execute enables billing operations. Keep the state file and
run only ONE copy/session; the filesystem lock is not a cross-machine lock.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

API = "https://compute.googleapis.com/compute/v1/projects/"
CAPACITY = {"ZONE_RESOURCE_POOL_EXHAUSTED", "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
            "RESOURCE_POOL_EXHAUSTED"}
PREFIX = "soup-t4-recovery"
DEFAULT_ZONES = """
asia-northeast1-c asia-northeast1-a
asia-east1-a asia-east1-c asia-east2-a asia-east2-c
asia-northeast3-b asia-northeast3-c asia-south1-a asia-south1-b
asia-southeast1-a asia-southeast1-b asia-southeast1-c
asia-southeast2-a asia-southeast2-b
australia-southeast1-a australia-southeast1-c
europe-central2-b europe-central2-c europe-west1-b europe-west1-c europe-west1-d
europe-west2-a europe-west2-b europe-west3-b europe-west4-a europe-west4-b europe-west4-c
me-west1-b me-west1-c northamerica-northeast1-c
southamerica-east1-a southamerica-east1-b southamerica-east1-c
us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c
us-east4-a us-east4-b us-east4-c us-west1-a us-west1-b us-west2-b us-west2-c
us-west3-b us-west4-a
""".split()


class Halt(RuntimeError):
    pass


class ApiError(Halt):
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload
        super().__init__(f"HTTP {status}: {json.dumps(payload)}")


class TransportError(Halt):
    pass


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


class Compute:
    def __init__(self, project):
        self.base = API + project + "/"
        self.token = None
        self.token_at = 0

    def call(self, method, path, body=None):
        if not self.token or time.monotonic() - self.token_at > 2400:
            p = subprocess.run(["gcloud", "auth", "print-access-token", "--quiet"],
                               capture_output=True, text=True, timeout=60)
            if p.returncode or not p.stdout.strip():
                raise Halt("gcloud authentication failed. Run gcloud auth login, then resume.")
            self.token, self.token_at = p.stdout.strip(), time.monotonic()
        req = urllib.request.Request(self.base + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.token,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"message": raw[:1500]}
            raise ApiError(exc.code, payload) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise TransportError(f"Uncertain response ({type(exc).__name__}); state retained.") from None


class State:
    def __init__(self, path, config):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
            if self.data.get("config") != config:
                raise Halt("Arguments differ from saved state. Resume with the same arguments; do not delete state to bypass an unresolved request.")
        else:
            self.data = {"config": config, "intent": None, "next_zone": 0, "winner": None}
            self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        if os.name == "posix":
            fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


@contextlib.contextmanager
def local_lock(path):
    import fcntl
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Halt("Another copy is running in this Cloud Shell. Stop here.") from None
        yield


def inventory(api):
    """Scan ALL zones, including those removed from a later command line."""
    found, page = [], ""
    while True:
        query = urllib.parse.urlencode({"maxResults": 500, **({"pageToken": page} if page else {})})
        result = api.call("GET", "aggregated/instances?" + query)
        if result.get("unreachables") or result.get("warning"):
            raise Halt("Incomplete project inventory. Refusing to create a VM.")
        for scope in result.get("items", {}).values():
            warning = scope.get("warning", {})
            if warning and warning.get("code") != "NO_RESULTS_ON_PAGE":
                raise Halt("Incomplete zonal inventory. Refusing to create a VM.")
            for vm in scope.get("instances", []):
                if vm["name"].startswith(PREFIX + "-"):
                    found.append(vm)
        page = result.get("nextPageToken")
        if not page:
            return found


def remember_existing(api, state):
    found = inventory(api)
    if len(found) > 1:
        raise Halt("Multiple recovery VMs already exist: " + ", ".join(v["name"] for v in found) + ". No further creation.")
    if not found:
        if state.data.get("winner"):
            raise Halt("Previously recorded VM is now absent. Refusing automatic replacement.")
        return False
    vm = found[0]
    zone = vm["zone"].rsplit("/", 1)[-1]
    if vm.get("labels", {}).get("recovery-tool") != "create-one-t4-v1":
        raise Halt("Name collision with existing VM " + vm["name"] + ". No changes made.")
    status = vm.get("status", "UNKNOWN")
    ip = next((a["natIP"] for n in vm.get("networkInterfaces", [])
               for a in n.get("accessConfigs", []) if a.get("natIP")), "not assigned")
    state.data["winner"] = {"name": vm["name"], "zone": zone, "status": status, "ip": ip}
    state.save()
    log(f"Existing/created VM: {vm['name']} | {zone} | {status} | IP {ip}")
    if status != "RUNNING":
        raise Halt("VM exists but is not RUNNING. Stopping here; no second VM will be created. Inspect this VM before resuming.")
    log("ONE VM RUNNING. GPU drivers and training runtime still need setup/verification.")
    print(f"gcloud compute ssh {vm['name']} --project={state.data['config']['project']} --zone={zone}")
    return True


def make_body(config, zone, image):
    project = config["project"]
    base = API + project
    name = PREFIX + "-" + zone
    init = {"diskName": name, "diskSizeGb": "100", "diskType": f"{base}/zones/{zone}/diskTypes/pd-balanced"}
    if config["snapshot"]:
        init["sourceSnapshot"] = config["snapshot"]
    else:
        init["sourceImage"] = image
    body = {
        "name": name,
        "machineType": f"{base}/zones/{zone}/machineTypes/n1-standard-8",
        "guestAccelerators": [{"acceleratorType": f"{base}/zones/{zone}/acceleratorTypes/nvidia-tesla-t4", "acceleratorCount": 1}],
        "disks": [{"boot": True, "autoDelete": False, "type": "PERSISTENT", "initializeParams": init}],
        "networkInterfaces": [{"network": f"{base}/global/networks/default",
                               "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}]}],
        "scheduling": {"provisioningModel": config["model"], "onHostMaintenance": "TERMINATE",
                       "automaticRestart": config["model"] == "STANDARD"},
        "labels": {"recovery-tool": "create-one-t4-v1"},
        "deletionProtection": True,
        "serviceAccounts": [],
    }
    if config["model"] == "SPOT":
        body["scheduling"]["instanceTerminationAction"] = "STOP"
    if config["service_account"]:
        body["serviceAccounts"] = [{"email": config["service_account"],
            "scopes": ["https://www.googleapis.com/auth/spreadsheets"]}]
    return body


def capacity_only(operation):
    errors = operation.get("error", {}).get("errors", [])
    return bool(errors) and all(e.get("code") in CAPACITY for e in errors)


def require_no_disk(api, zone, name):


    try:
        api.call("GET", f"zones/{zone}/disks/{name}")
    except ApiError as exc:
        if exc.status == 404:
            return
        raise
    raise Halt(f"Disk {name} exists in {zone}. Inspect it before retrying; no additional resources will be created.")


def finish_intent(api, state, sleep=time.sleep):
    """No new zone or new request ID until this intent has a definite outcome."""
    intent = state.data["intent"]
    zone = intent["zone"]
    operation = intent.get("operation")
    if not operation:

        if time.time() - intent["created_at"] > 3600:
            raise Halt("Unresolved request is over one hour old. Inspect Compute operations before proceeding; automatic replay disabled.")
        path = f"zones/{zone}/instances?requestId={intent['request_id']}"
        for attempt in range(3):
            try:
                operation = api.call("POST", path, intent["body"])
                break
            except (TransportError, ApiError) as exc:
                if isinstance(exc, ApiError) and exc.status < 500:
                    raise
                if attempt == 2:
                    raise Halt("Create outcome remains uncertain. Resume the SAME command/state to reconcile; do not run another creator.") from exc
                log("Uncertain create response; retrying the SAME request ID only.")
                sleep(10)
        intent["operation"] = operation
        state.save()
    if not operation.get("name"):
        raise Halt("Create did not return an operation name. State retained for inspection.")
    poll_until = time.monotonic() + 900
    while operation.get("status") != "DONE":
        if time.monotonic() > poll_until:
            raise Halt("Operation still pending after 15 minutes. State retained; resume the same command to continue polling.")
        sleep(5)
        operation = api.call("GET", f"zones/{zone}/operations/{operation['name']}")
    intent["operation"] = operation
    state.save()
    if operation.get("error"):
        if not capacity_only(operation):
            raise Halt("Create failed; no other zone attempted: " + json.dumps(operation["error"]))

        if remember_existing(api, state):
            return True
        require_no_disk(api, zone, intent["body"]["name"])
        log(f"{zone}: capacity exhausted. Safe to try the next zone.")
        state.data["intent"] = None
        state.data["next_zone"] = (state.data["next_zone"] + 1) % len(state.data["config"]["zones"])
        state.save()
        return False
    for _ in range(12):

        vm = api.call("GET", f"zones/{zone}/instances/{intent['body']['name']}")
        if vm.get("status") == "RUNNING":
            state.data["winner"] = {"name": vm["name"], "zone": zone}
            state.save()
            if remember_existing(api, state):
                return True
        if vm.get("status") in {"TERMINATED", "STOPPING", "SUSPENDED", "SUSPENDING"}:
            raise Halt(f"Created VM is {vm['status']}; no second VM will be created.")
        sleep(5)
    raise Halt("Create succeeded, but VM is not yet RUNNING. No second VM will be created; resume to inspect.")


def run(api, state, interval, max_minutes):
    if remember_existing(api, state):
        return
    config = state.data["config"]

    if state.data["intent"] and finish_intent(api, state):
        return
    api.call("GET", "global/networks/default")
    if config["snapshot"]:
        image = None
    else:
        image_api = Compute("ubuntu-os-cloud")
        image = image_api.call("GET", "global/images/family/ubuntu-2204-lts")["selfLink"]
    deadline = time.monotonic() + max_minutes * 60 if max_minutes else float("inf")
    while time.monotonic() < deadline:
        if remember_existing(api, state):
            return
        zone = config["zones"][state.data["next_zone"]]
        api.call("GET", f"zones/{zone}/acceleratorTypes/nvidia-tesla-t4")
        api.call("GET", f"zones/{zone}/machineTypes/n1-standard-8")
        require_no_disk(api, zone, PREFIX + "-" + zone)
        state.data["intent"] = {"zone": zone, "request_id": str(uuid.uuid4()),
            "created_at": time.time(), "body": make_body(config, zone, image)}
        state.save()
        log(f"Requesting one {config['model']} n1-standard-8 + T4 in {zone}.")
        if finish_intent(api, state):
            return
        time.sleep(interval)
    log("Retry deadline reached. All attempted requests had definitive capacity failures; nothing else submitted.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="project-ceb435db-d81b-4fa9-b28")
    p.add_argument("--zones", nargs="+", default=DEFAULT_ZONES)
    p.add_argument("--model", choices=["SPOT", "STANDARD"], default="SPOT")
    p.add_argument("--interval", type=int, default=30, help="Seconds between definite capacity failures (minimum 10)")
    p.add_argument("--max-minutes", type=int, default=0, help="0 retries until success/Ctrl+C; unresolved operations always reconcile first")
    p.add_argument("--service-account", default="", help="Optional dedicated account; grants Sheets OAuth scope, not IAM permissions")
    p.add_argument("--snapshot", default="", help="Optional existing snapshot selfLink for a COPY of the old disk, otherwise fresh Ubuntu")
    p.add_argument("--execute", action="store_true", help="Actually create billable resources")
    args = p.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project):
        p.error("Invalid project ID")
    if any(not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+-[a-z]", z) for z in args.zones):
        p.error("Invalid zone name")
    if args.interval < 10 or args.max_minutes < 0:
        p.error("interval must be >=10 and max-minutes >=0")
    if args.snapshot and not re.fullmatch(r"https://(?:www.googleapis.com|compute.googleapis.com)/compute/v1/projects/[a-z0-9-]+/global/snapshots/[a-z0-9-]+", args.snapshot):
        p.error("snapshot must be a full Compute v1 snapshot selfLink")
    config = {"project": args.project, "zones": list(dict.fromkeys(args.zones)), "model": args.model,
              "snapshot": args.snapshot, "service_account": args.service_account}
    if not args.execute:
        print("PREVIEW ONLY. Add --execute to create billable resources. Run only one Cloud Shell copy.")
        print(json.dumps(config, indent=2))
        print(json.dumps(make_body(config, config["zones"][0], "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"), indent=2))
        return
    root = Path.home() / ".soup-t4-recovery" / args.project
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with local_lock(root / "creator.lock"):
        state = State(root / "state.json", config)
        log(f"State: {state.path}. Retain this file. Ctrl+C stops retries, not an in-flight cloud operation.")
        log("Billing starts on creation. Spot may stop at any time. Existing VM/disk/IP will not be modified.")
        run(Compute(args.project), state, args.interval, args.max_minutes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. State retained. Resume the same command to reconcile any pending request.", file=sys.stderr)
        sys.exit(130)
    except (Halt, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
