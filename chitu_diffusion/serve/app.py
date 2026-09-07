from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, HTTPException, Response, status

from .protocol import (
    AdmissionResponse,
    CancelResponse,
    HealthResponse,
    ImageGenerateRequest,
    RequestStatusResponse,
)


class ServiceBackend(Protocol):
    def submit(self, request: ImageGenerateRequest) -> AdmissionResponse: ...

    def status(self, request_id: str) -> RequestStatusResponse | None: ...

    def cancel(self, request_id: str) -> CancelResponse | None: ...

    def image(self, request_id: str) -> bytes | None: ...

    def media_type(self, request_id: str) -> str | None: ...

    def health(self) -> HealthResponse: ...


def create_app(backend: ServiceBackend) -> FastAPI:
    app = FastAPI(title="Chitu Diffusion Service", version="0.1.0")

    @app.post(
        "/v1/image-decode",
        response_model=AdmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit(request: ImageGenerateRequest) -> AdmissionResponse:
        try:
            return backend.submit(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/image-decode/{request_id}", response_model=RequestStatusResponse)
    def request_status(request_id: str) -> RequestStatusResponse:
        result = backend.status(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail="request not found")
        return result

    def _media_response(request_id: str, *, label: str) -> Response:
        request_status = backend.status(request_id)
        if request_status is None:
            raise HTTPException(status_code=404, detail="request not found")
        if request_status.status != "completed":
            raise HTTPException(status_code=409, detail=f"{label} is not ready")
        get_media = getattr(backend, "media", backend.image)
        payload = get_media(request_id)
        if payload is None:
            raise HTTPException(
                status_code=500, detail=f"completed request has no {label}"
            )
        return Response(
            content=payload,
            media_type=backend.media_type(request_id) or "application/octet-stream",
        )

    @app.get("/v1/media/{request_id}")
    def request_media(request_id: str) -> Response:
        return _media_response(request_id, label="media")

    @app.get("/v1/image-decode/{request_id}/image")
    def request_image(request_id: str) -> Response:
        return _media_response(request_id, label="image")

    @app.delete("/v1/image-decode/{request_id}", response_model=CancelResponse)
    def cancel(request_id: str) -> CancelResponse:
        result = backend.cancel(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail="request not found")
        if not result.cancelled:
            raise HTTPException(status_code=409, detail=result.detail)
        return result

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        response = backend.health()
        if response.state == "failed":
            raise HTTPException(
                status_code=503, detail=response.fatal_error or "runtime failed"
            )
        return response

    return app
