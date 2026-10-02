#!/usr/bin/env python3
"""Small least-inflight reverse proxy for equivalent OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from dataclasses import dataclass
from typing import Iterable

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


@dataclass
class Backend:
    base_url: str
    inflight: int = 0
    requests: int = 0
    failures: int = 0


class LoadBalancer:
    def __init__(self, backends: Iterable[str], timeout: float) -> None:
        self.backends = [Backend(value.rstrip("/")) for value in backends]
        self.timeout = timeout
        self._next = 0
        self._lock = asyncio.Lock()
        self.session: ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(
            timeout=ClientTimeout(total=self.timeout),
            connector=TCPConnector(limit=0, ttl_dns_cache=300),
        )

    async def stop(self, _app: web.Application) -> None:
        if self.session is not None:
            await self.session.close()

    async def select(self) -> Backend:
        async with self._lock:
            minimum = min(item.inflight for item in self.backends)
            eligible = {
                index
                for index, item in enumerate(self.backends)
                if item.inflight == minimum
            }
            for offset in range(len(self.backends)):
                index = (self._next + offset) % len(self.backends)
                if index in eligible:
                    self._next = (index + 1) % len(self.backends)
                    backend = self.backends[index]
                    backend.inflight += 1
                    backend.requests += 1
                    return backend
        raise RuntimeError("No backend available")

    async def release(self, backend: Backend, failed: bool) -> None:
        async with self._lock:
            backend.inflight -= 1
            if failed:
                backend.failures += 1

    async def proxy(self, request: web.Request) -> web.Response:
        if request.path == "/_lb/status":
            return web.json_response(
                {
                    "backends": [
                        {
                            "url": item.base_url,
                            "inflight": item.inflight,
                            "requests": item.requests,
                            "failures": item.failures,
                        }
                        for item in self.backends
                    ]
                }
            )
        backend = await self.select()
        failed = False
        try:
            assert self.session is not None
            suffix = request.path_qs
            url = backend.base_url + (suffix if suffix.startswith("/") else "/" + suffix)
            body = await request.read()
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower() not in HOP_BY_HOP
                and key.lower() not in {"host", "content-length", "accept-encoding"}
            }
            async with self.session.request(
                request.method, url, data=body, headers=headers
            ) as upstream:
                payload = await upstream.read()
                response_headers = {
                    key: value
                    for key, value in upstream.headers.items()
                    if key.lower() not in HOP_BY_HOP
                    and key.lower() not in {"content-length", "content-encoding"}
                }
                failed = upstream.status >= 500
                return web.Response(
                    status=upstream.status,
                    body=payload,
                    headers=response_headers,
                )
        except Exception as exc:  # noqa: BLE001
            failed = True
            return web.json_response(
                {"error": {"message": f"load balancer upstream error: {exc}"}},
                status=502,
            )
        finally:
            await self.release(backend, failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    balancer = LoadBalancer(args.backend, args.timeout)
    app = web.Application(client_max_size=64 * 1024**2)
    app.router.add_route("*", "/{path:.*}", balancer.proxy)
    app.on_startup.append(balancer.start)
    app.on_cleanup.append(balancer.stop)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
