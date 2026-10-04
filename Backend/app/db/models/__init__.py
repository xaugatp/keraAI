"""Importing this package registers every ORM model on ``Base.metadata``.

Alembic's env.py (and tests that call ``create_all``) import it for that side
effect — add each new model module here so autogenerate can see it.
"""

from app.db.models.analysis import Analysis

__all__ = ["Analysis"]
