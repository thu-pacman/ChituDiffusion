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

    def health(self) -> HealthResponse: ...


def create_app(backend: ServiceBackend) -> FastAPI:
    app = FastAPI(title="EPAC Z-Image Service", version="0.1.0")

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

    @app.get("/v1/image-decode/{request_id}/image")
    def request_image(request_id: str) -> Response:
        request_status = backend.status(request_id)
        if request_status is None:
            raise HTTPException(status_code=404, detail="request not found")
        if request_status.status != "completed":
            raise HTTPException(status_code=409, detail="image is not ready")
        image = backend.image(request_id)
        if image is None:
            raise HTTPException(
                status_code=500, detail="completed request has no image"
            )
        return Response(content=image, media_type="image/png")

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
