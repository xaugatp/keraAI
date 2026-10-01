from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import SettingsDep


def test_settings_dep_resolves_to_real_settings(app: FastAPI) -> None:
    @app.get("/__test__/settings-dep")
    async def _read_app_name(settings: SettingsDep) -> dict[str, str]:
        return {"app_name": settings.app_name}

    with TestClient(app) as client:
        response = client.get("/__test__/settings-dep")

    assert response.status_code == 200
    assert response.json() == {"app_name": "KeraAI API"}
