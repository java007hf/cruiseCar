# CruiseCar Python server image.
# Pure standard-library Python app: no pip install step required.
FROM python:3.11-slim

# Run as a non-root user for a smaller attack surface.
RUN groupadd --system cruisecar && useradd --system --gid cruisecar --create-home cruisecar

# The Python package imports are relative to the `server/` directory
# (e.g. `control_server.server`, `manager_api.config.settings`), so the
# working directory MUST be /app/server.
WORKDIR /app/server

# Only copy the Python and static web sources; keep the SQLite DB on a mounted volume.
# Build context is the repository root; see server/docker-compose.yml.
COPY server/control_server/ ./control_server/
COPY server/manager_api/     ./manager_api/
COPY server/manager_web/     ./manager_web/
COPY web_send/ /app/web_send/

# Default deployment; override at runtime with `docker run -e CRUISECAR_DEPLOYMENT=full`.
ENV CRUISECAR_DEPLOYMENT=light \
    CRUISECAR_HOST=0.0.0.0 \
    CRUISECAR_DB=/data/cruisecar.db

# Light mode ports. Add 8088/8089 when running full mode.
EXPOSE 42110 42112 8088 8089

USER cruisecar

CMD ["python3", "-m", "control_server.server"]
