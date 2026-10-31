#!/usr/bin/env python3
"""Render non-secret deployment templates into a NEW review directory; never install/start them."""
import argparse,ipaddress
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-ip',required=True);p.add_argument('--out',required=True);a=p.parse_args();ip=ipaddress.ip_address(a.public_ip)
if ip.version!=4 or not ip.is_global:raise SystemExit('This template requires the verified globally routable IPv4 address')
root=Path(__file__).resolve().parent;out=Path(a.out).absolute();out.mkdir(parents=False,exist_ok=False)
for name in ('nginx_acme_only.conf.template','nginx_ip.conf.template','soup-api.service.template'):
 (out/name.removesuffix('.template')).write_text((root/name).read_text().replace('__PUBLIC_IP__',str(ip)))
print('Rendered review-only files. No service, firewall, certificate or network changes performed.')
