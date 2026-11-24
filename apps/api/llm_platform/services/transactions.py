"""Compose existing domain operations in an outer orchestration transaction."""

from contextlib import contextmanager

from sqlalchemy.orm import Session


def commit_or_flush(session: Session) -> None:
    if session.info.get("domain_transaction_depth", 0):
        session.flush()
    else:
        session.commit()


@contextmanager
def domain_transaction(session: Session):
    depth = session.info.get("domain_transaction_depth", 0)
    session.info["domain_transaction_depth"] = depth + 1
    try:
        yield
        session.flush()
    finally:
        session.info["domain_transaction_depth"] = depth
