#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE_URI="${1:?Usage: deploy-ec2.sh <image-uri>}"
ENV_FILE="/etc/aerointel/production.env"
CADDYFILE="/etc/caddy/Caddyfile"
CADDY_BACKUP="/etc/caddy/Caddyfile.aerointel-backup"
CANDIDATE="aerointel-api-candidate"
PREVIOUS="aerointel-api-previous"
ACTIVE="aerointel-api"
REGION="ap-southeast-2"

if [[ ! -r "$ENV_FILE" ]]; then
  echo "Production environment file is missing or unreadable."
  exit 1
fi

if [[ ! -f "$CADDYFILE" ]]; then
  echo "Caddyfile not found."
  exit 1
fi

ACTIVE_PORT="$(awk '/reverse_proxy[[:space:]]/ { split($2, a, ":"); print a[length(a)] }' "$CADDYFILE")"
case "$ACTIVE_PORT" in
  8000) CANDIDATE_PORT=8001 ;;
  8001) CANDIDATE_PORT=8000 ;;
  *) echo "Unexpected active Caddy upstream port: $ACTIVE_PORT"; exit 1 ;;
esac

echo "Logging in to private ECR..."
aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "${IMAGE_URI%%/*}"

echo "Pulling image: $IMAGE_URI"
docker pull "$IMAGE_URI"

# Remove only the container retained from the previous successful deployment.
# The currently active aerointel-api container is never removed here.
if docker container inspect "$PREVIOUS" >/dev/null 2>&1; then
  docker rm -f "$PREVIOUS"
fi
if docker container inspect "$CANDIDATE" >/dev/null 2>&1; then
  docker rm -f "$CANDIDATE"
fi

echo "Applying database migrations before starting the candidate..."
docker run --rm --env-file "$ENV_FILE" -e RUN_MIGRATIONS=true \
  "$IMAGE_URI" sh -c "alembic upgrade head"

echo "Starting candidate on 127.0.0.1:$CANDIDATE_PORT..."
docker run -d \
  --name "$CANDIDATE" \
  --restart unless-stopped \
  --init \
  --env-file "$ENV_FILE" \
  -e RUN_MIGRATIONS=false \
  -p "127.0.0.1:$CANDIDATE_PORT:8000" \
  --security-opt no-new-privileges:true \
  --stop-timeout 30 \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  "$IMAGE_URI" >/dev/null

echo "Waiting for application readiness..."
READY=0
for attempt in $(seq 1 72); do
  if curl --silent --show-error --fail \
      "http://127.0.0.1:$CANDIDATE_PORT/ready" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "Candidate did not become ready. Active production container was not changed."
  docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
  exit 1
fi

echo "Candidate is ready. Validating Caddy configuration..."
cp "$CADDYFILE" "$CADDY_BACKUP"
sed "s#127.0.0.1:$ACTIVE_PORT#127.0.0.1:$CANDIDATE_PORT#g" \
  "$CADDY_BACKUP" > /etc/caddy/Caddyfile.aerointel-candidate

if ! caddy validate --config /etc/caddy/Caddyfile.aerointel-candidate; then
  rm -f /etc/caddy/Caddyfile.aerointel-candidate
  docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
  echo "Caddy validation failed. Existing production route is unchanged."
  exit 1
fi

install -o root -g root -m 0644 \
  /etc/caddy/Caddyfile.aerointel-candidate "$CADDYFILE"
rm -f /etc/caddy/Caddyfile.aerointel-candidate

if ! systemctl reload caddy; then
  install -o root -g root -m 0644 "$CADDY_BACKUP" "$CADDYFILE"
  systemctl reload caddy || true
  docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
  echo "Caddy reload failed. Previous route restored."
  exit 1
fi

# Keep the old container running on its previous loopback port for rollback.
if docker container inspect "$PREVIOUS" >/dev/null 2>&1; then
  docker rm -f "$PREVIOUS"
fi
docker rename "$ACTIVE" "$PREVIOUS"
docker rename "$CANDIDATE" "$ACTIVE"

echo "Deployment successful."
echo "Active image: $IMAGE_URI"
echo "Active upstream: 127.0.0.1:$CANDIDATE_PORT"
echo "Previous container retained as: $PREVIOUS"
