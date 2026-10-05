# 0009 — SQL Server access via Windows authentication on this laptop (resolves O-01)

**Status:** accepted · **Date:** 2026-10-05

## Context
The owner's SQL Server instance has Mixed Mode off (Windows Authentication only) and TCP/IP disabled for the default
instance (see the comments in `.env.example`). The spec's default was a SQL login `kera_app`.

## Decision
The app connects with `DB_TRUSTED_CONNECTION=true` (Windows auth) over ODBC Driver 18.
`db/sql/001_create_database_and_login.sql` supports both: it grants the roles to the Windows login via `@WindowsLogin`
(skipped if that login is already a sysadmin) and only creates `kera_app` once a non-placeholder password is set, so a
login with a known password is never created.

## Consequences
+ No secret to store; works today.
− The API process must run as an account with access (the owner's). A service account is needed if the app is ever run
  as a Windows service. `db_ddladmin` is a dev-only convenience (migrations and app share one login).
