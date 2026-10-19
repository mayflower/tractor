FROM ghcr.io/astral-sh/uv@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS build
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl build-essential autoconf automake libtool pkg-config && rm -rf /var/lib/apt/lists/*
# Same library revision and frozen data as the Nix development environment.
RUN git clone https://github.com/openvenues/libpostal.git /tmp/libpostal && \
    cd /tmp/libpostal && git checkout ffafa3fcd687631955453df70906c53d5677ce9a && \
    ./bootstrap.sh && ./configure --disable-data-download --datadir=/opt && \
    make -j2 && make install && ldconfig && rm -rf /tmp/libpostal
COPY configs/libpostal-data.sha256 /tmp/libpostal-data.sha256
RUN mkdir -p /opt/libpostal && \
    for name in libpostal_data parser language_classifier; do \
      curl -fL --retry 3 "https://github.com/openvenues/libpostal/releases/download/v1.0.0/${name}.tar.gz" -o /tmp/postal.tar.gz && \
      tar -xzf /tmp/postal.tar.gz -C /opt/libpostal || exit 1; \
    done && cd /opt/libpostal && sha256sum -c /tmp/libpostal-data.sha256 && rm /tmp/postal.tar.gz
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra server --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --extra server --no-dev --no-editable

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 tractor
COPY --from=build /app/.venv /app/.venv
COPY --from=build /usr/local/lib/libpostal.so* /usr/local/lib/
COPY --from=build /opt/libpostal /opt/libpostal
RUN ldconfig
ENV PATH="/app/.venv/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
USER tractor
WORKDIR /home/tractor
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=5)"
ENTRYPOINT ["tractor", "serve"]
CMD ["--model", "/model", "--device", "cpu", "--host", "0.0.0.0", "--port", "8000"]
