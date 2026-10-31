# Temporary HTTPS relay

As verified on **2026-10-31**, the alternate Soup endpoint is:

**https://shareware-draft-optional-feeding.trycloudflare.com**

This is a Cloudflare reverse proxy to the existing Soup service. It is not a general-purpose forward proxy: do not set `HTTPS_PROXY` to this URL. No new proxy environment setting or proxy password is required. The existing [IP endpoint](https://35.200.1.242/) and [hostname endpoint](https://35-200-1-242.sslip.io/) remain available.

## Client check and use

First check connectivity without sending a token:

```bash
curl --connect-timeout 10 --max-time 20 \
  --silent --show-error --output /dev/null \
  --write-out 'HTTP %{http_code}\n' \
  https://shareware-draft-optional-feeding.trycloudflare.com/v1/notebook-runtime
```

**HTTP 401 is the expected successful connectivity result**: TLS and routing worked, and the API requires authentication. Keep certificate verification enabled; do not add `-k` or disable verification in Python.

Use the existing client and token, supplied privately through `SOUP_NOTEBOOK_TOKEN` or its hidden prompt. Put `--url` **before** the subcommand:

```bash
python notebook_client.py --url https://shareware-draft-optional-feeding.trycloudflare.com runtime
python notebook_client.py --url https://shareware-draft-optional-feeding.trycloudflare.com list
```

Authentication, jobs, results, and the 80-second training limit are unchanged. Cloudflare terminates client HTTPS and processes request headers, including the bearer token, and request/response bodies. The relay also verifies TLS to the existing Nginx origin; verification must remain enabled on both legs. Never put tokens in URLs, notebooks, logs, or shared examples.

The client still needs permitted DNS and outbound HTTPS access to Cloudflare. This relay does not enable networking in a restricted execution environment. A successful test elsewhere does not establish that a blocked runner can connect.

## Operator notes

- Systemd service: `soup-https-relay`. It uses `Restart=no` and is **not enabled at boot**.
- Binary: `/usr/local/bin/cloudflared-soup`, version `2026.10.7`.
- Binary SHA-256: `1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872`.
- Origin: existing HTTPS Nginx, with the existing certificate name used for TLS verification. Do not enable `noTLSVerify`.
- Nginx's existing DNS-origin map maps only the exact relay origin `https://shareware-draft-optional-feeding.trycloudflare.com` to `https://35.200.1.242`; unrelated origins remain subject to rejection. `map_hash_bucket_size` is `128`.
- Keep relay logs at `info`, not `debug`: debug logging can expose authorization headers. Keep the metrics listener local.

The deployed unit is recorded in [deploy/soup-https-relay.service.example](deploy/soup-https-relay.service.example). It runs without root privileges, caps memory at 256 MiB, and uses host CPUs 4–7, outside the notebook's 0–3 affinity.

Inspect the current process and URL with:

```bash
sudo systemctl status soup-https-relay --no-pager
sudo journalctl -u soup-https-relay -n 60 --no-pager
```

Starting a new tunnel, including a manual service restart, creates a **new hostname**. Read the new URL from its logs, replace the exact relay origin in the Nginx DNS-origin map, run `sudo nginx -t`, then reload Nginx with `sudo systemctl reload nginx`. Never wildcard-trust `*.trycloudflare.com` or remove Origin validation. Update this document and share the new URL only after fresh checks: token-free runtime returns 401, authenticated runtime returns 200, the exact relay Origin works, foreign Origins are rejected, and both original endpoints still work. Do not restart the API or GPU worker for a relay change.

To remove the route, stop only the relay:

```bash
sudo systemctl stop soup-https-relay
```

Then remove only its exact Origin mapping from the Nginx DNS-origin map, validate with `sudo nginx -t`, and reload Nginx. Leave the original IP/hostname mappings intact. No API or worker restart is required.

Quick Tunnels have no uptime guarantee, change hostname on each start, support at most 200 in-flight requests, and do not support SSE. Use the client's ordinary status polling. For a durable address, provision a named Cloudflare Tunnel with an account and domain.

Official references: [Quick Tunnels and limits](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [origin TLS and Host settings](https://developers.cloudflare.com/tunnel/reference/origin-parameters/), [logging parameters](https://developers.cloudflare.com/tunnel/reference/run-parameters/), [connector network prerequisites](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/troubleshoot-tunnels/connectivity-prechecks/).
