
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

DOCKER_NETWORK="aerointel-net"
REDIS_CONTAINER="aerointel-redis"
REDIS_VOLUME="aerointel-redis-data"
REDIS_IMAGE="redis:7-alpine"
REDIS_URL="redis://${REDIS_CONTAINER}:6379/0"

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

echo "Pulling application image..."
docker pull "$IMAGE_URI"

echo "Creating private Docker network if required..."

docker network inspect "$DOCKER_NETWORK" >/dev/null 2>&1 ||
  docker network create "$DOCKER_NETWORK" >/dev/null

echo "Creating persistent Redis volume if required..."

docker volume inspect "$REDIS_VOLUME" >/dev/null 2>&1 ||
  docker volume create "$REDIS_VOLUME" >/dev/null

echo "Starting persistent Redis service..."

docker pull "$REDIS_IMAGE"

if ! docker container inspect "$REDIS_CONTAINER" >/dev/null 2>&1; then
  docker run -d \
    --name "$REDIS_CONTAINER" \
    --network "$DOCKER_NETWORK" \
    --restart unless-stopped \
    --init \
    --security-opt no-new-privileges:true \
    --memory 512m \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    -v "$REDIS_VOLUME:/data" \
    "$REDIS_IMAGE" \
    redis-server --appendonly yes >/dev/null
else
  docker network connect "$DOCKER_NETWORK" "$REDIS_CONTAINER" \
    >/dev/null 2>&1 || true

  docker start "$REDIS_CONTAINER" >/dev/null 2>&1 || true
fi

echo "Checking Redis availability..."

REDIS_READY=0

for attempt in $(seq 1 20); do
  if docker exec "$REDIS_CONTAINER" redis-cli ping 2>/dev/null |
      grep -q PONG; then
    REDIS_READY=1
    break
  fi

  sleep 2
done

if [[ "$REDIS_READY" -ne 1 ]]; then
  echo "Redis is unavailable. Deployment aborted."
  exit 1
fi

echo "Redis is ready."

# Remove only containers retained from previous deployment.
if docker container inspect "$PREVIOUS" >/dev/null 2>&1; then
  docker rm -f "$PREVIOUS"
fi

if docker container inspect "$CANDIDATE" >/dev/null 2>&1; then
  docker rm -f "$CANDIDATE"
fi

echo "Applying database migrations..."

docker run --rm \
  --network "$DOCKER_NETWORK" \
  --env-file "$ENV_FILE" \
  -e RUN_MIGRATIONS=true \
  -e REDIS_URL="$REDIS_URL" \
  "$IMAGE_URI" \
  sh -c "alembic upgrade head"

echo "Starting candidate application..."

docker run -d \
  --name "$CANDIDATE" \
  --network "$DOCKER_NETWORK" \
  --restart unless-stopped \
  --init \
  --env-file "$ENV_FILE" \
  -e REDIS_URL="$REDIS_URL" \
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

if ! caddy validate \
    --config /etc/caddy/Caddyfile.aerointel-candidate \
    --adapter caddyfile; then

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

# Retain the previous application for rollback.
if docker container inspect "$PREVIOUS" >/dev/null 2>&1; then
  docker rm -f "$PREVIOUS"
fi

docker rename "$ACTIVE" "$PREVIOUS"
docker rename "$CANDIDATE" "$ACTIVE"

echo "========================================"
echo "Deployment successful."
echo "Active image: $IMAGE_URI"
echo "Active upstream: 127.0.0.1:$CANDIDATE_PORT"
echo "Redis container: $REDIS_CONTAINER"
echo "Redis volume: $REDIS_VOLUME"
echo "Previous container: $PREVIOUS"
echo "========================================"
