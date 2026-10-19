#!/usr/bin/env bash
# --recreate also checks named-volume persistence; it briefly stops these services.
set -euo pipefail
if [[ $# -gt 1 || ( $# == 1 && $1 != --recreate ) ]]; then
    echo 'Usage: bash scripts/check_local_services.sh [--recreate]' >&2
    exit 1
fi
cd "$(dirname "$0")/.."
compose=(docker compose --env-file .env -f infra/compose.yml)
token="vf_smoke_$(date +%s)_${RANDOM}"

sql() {
    "${compose[@]}" exec -T postgres sh -ec \
        'export PGPASSWORD="$POSTGRES_PASSWORD"; exec psql -h postgres -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -At "$@"' sh "$@"
}
redis() {
    "${compose[@]}" exec -T redis sh -ec \
        'export REDISCLI_AUTH="$REDIS_PASSWORD"; exec redis-cli --raw "$@"' sh "$@"
}
storage() {
    "${compose[@]}" exec -T minio sh -ec '
        config=$(mktemp -d)
        trap '\''rm -rf "$config"'\'' EXIT
        mc --config-dir "$config" alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
        mc --config-dir "$config" "$@"
    ' sh "$@"
}
cleanup() {
    local result=$?
    trap - EXIT
    sql -c "DROP SCHEMA IF EXISTS $token CASCADE" >/dev/null 2>&1 || echo "Could not remove SQL probe $token" >&2
    redis DEL "$token" >/dev/null || echo "Could not remove Redis probe $token" >&2
    storage rm "local/vf-artifacts/$token.txt" >/dev/null || echo "Could not remove object probe $token" >&2
    exit "$result"
}
trap cleanup EXIT

sql -c "CREATE SCHEMA $token; CREATE TABLE $token.probe (value text); INSERT INTO $token.probe VALUES ('$token')" >/dev/null
[[ $(redis SET "$token" "$token" EX 600) == OK ]]
printf '%s' "$token" | storage pipe "local/vf-artifacts/$token.txt" >/dev/null
for bucket in vf-images vf-datasets vf-artifacts vf-predictions; do
    storage stat "local/$bucket" >/dev/null
done

if [[ ${1:-} == --recreate ]]; then
    "${compose[@]}" up -d --force-recreate --wait --wait-timeout 120 postgres redis minio
fi

[[ $(sql -c "SELECT value FROM $token.probe") == "$token" ]]
[[ $(redis GET "$token") == "$token" ]]
[[ $(storage cat "local/vf-artifacts/$token.txt") == "$token" ]]
echo 'PostgreSQL, Redis and S3 authenticated read/write checks passed.'

if "${compose[@]}" exec -T postgres sh -ec \
    'PGPASSWORD=invalid psql -h postgres -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1"' >/dev/null 2>&1; then
    echo 'PostgreSQL accepted an incorrect password.' >&2
    exit 1
fi
[[ $("${compose[@]}" exec -T redis redis-cli ping) == NOAUTH* ]]
endpoint=$("${compose[@]}" port minio 9000)
[[ $(curl --silent --show-error --max-time 10 -o /dev/null -w '%{http_code}' "http://$endpoint/vf-artifacts/$token.txt") == 403 ]]
echo 'Incorrect PostgreSQL credentials and unauthenticated Redis/S3 requests were rejected.'
if [[ ${1:-} == --recreate ]]; then
    echo 'All three probes survived container recreation with named volumes.'
fi
