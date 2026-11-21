#!/usr/bin/env bash
# Run in Cloud Shell only after recovery health checks pass.
# Moves the existing reserved IP. Does not stop, reboot or delete either VM.
set -euo pipefail
project=project-ceb435db-d81b-4fa9-b28
zone=asia-northeast1-c
region=asia-northeast1
old=instance-20261030-172714
new=redacted-machine
address=35.200.1.242

[[ "$(gcloud compute instances describe "$old" --project="$project" --zone="$zone" --format='value(status)')" == TERMINATED ]]
[[ "$(gcloud compute instances describe "$new" --project="$project" --zone="$zone" --format='value(status)')" == RUNNING ]]
[[ "$(gcloud compute addresses describe soup-queue-existing-ip --project="$project" --region="$region" --format='value(address)')" == "$address" ]]

gcloud compute instances add-tags "$new" --project="$project" --zone="$zone" --tags=soup-queue-web

new_ip="$(gcloud compute instances describe "$new" --project="$project" --zone="$zone" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
if [[ "$new_ip" == "$address" ]]; then
  printf 'Recovery VM already owns %s. No IP changes needed.\n' "$address"
  exit 0
fi

old_ip="$(gcloud compute instances describe "$old" --project="$project" --zone="$zone" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
if [[ -n "$old_ip" && "$old_ip" != "$address" ]]; then
  printf 'Unexpected original VM IP: %s. Stopping.\n' "$old_ip" >&2
  exit 1
fi
if [[ -n "$old_ip" ]]; then
  old_access="$(gcloud compute instances describe "$old" --project="$project" --zone="$zone" --format='value(networkInterfaces[0].accessConfigs[0].name)')"
  gcloud compute instances delete-access-config "$old" --project="$project" --zone="$zone" --network-interface=nic0 --access-config-name="$old_access"
fi

if [[ -n "$new_ip" ]]; then
  new_access="$(gcloud compute instances describe "$new" --project="$project" --zone="$zone" --format='value(networkInterfaces[0].accessConfigs[0].name)')"
  gcloud compute instances delete-access-config "$new" --project="$project" --zone="$zone" --network-interface=nic0 --access-config-name="$new_access"
fi

gcloud compute instances add-access-config "$new" --project="$project" --zone="$zone" --network-interface=nic0 --access-config-name='External NAT' --address="$address"
[[ "$(gcloud compute instances describe "$new" --project="$project" --zone="$zone" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')" == "$address" ]]
printf 'Public IP restored: https://%s/\n' "$address"
