# 0001 — Synchronous SQLAlchemy + pyodbc, blocking work in the threadpool (P-01)

**Status:** accepted (spec P-01) · **Date:** 2026-10-05

## Context
SQL Server is reached through `pyodbc`, which is synchronous. Async MSSQL drivers are less mature.
Inference (torch) and disk I/O are blocking too. FastAPI routes are `async def`.

## Decision
Use sync SQLAlchemy 2.0 + pyodbc. Routes stay `async`, read the upload, then call the service via
`run_in_threadpool`, so inference, DB and disk work never block the event loop (and `/health` stays responsive).
Each request gets its own `Session` (`get_db`: the service commits, the dependency rolls back and closes in `finally`).

## Consequences
+ Mature, simple stack; one mental model.
− Concurrency is bounded by the threadpool and the single model lock (ADR 0003).
An async driver can replace this later without touching routers (layering: api → services → repositories → db).
