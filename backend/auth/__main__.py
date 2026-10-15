"""Manage login accounts (stored in LM_AUTH_DB_PATH, default storage/auth.db).

    py -m backend.auth add-user alice --role admin            # asks for the password twice (hidden)
    py -m backend.auth set-password alice                     # also ends all of alice's sessions
    py -m backend.auth set-role alice operator | disable alice | enable alice | unlock alice | list

Passwords are never accepted as command arguments (they would end up in shell history and process lists).
For automation, pipe one line on standard input with --password-stdin. Roles: viewer, operator, admin.
Accounts can be prepared while LM_AUTH_MODE=off; they are only used once login is turned on.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import sys

from backend.auth import build_service
from backend.auth.passwords import PasswordPolicyError
from backend.auth.store import ROLES
from backend.core.config import AppConfig


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("New password: ")
    if getpass.getpass("Repeat password: ") != first:
        raise PasswordPolicyError("the two passwords do not match")
    return first


def _fmt(t: float | None) -> str:
    return dt.datetime.fromtimestamp(t).isoformat(timespec="seconds") if t else "-"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="py -m backend.auth", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-user")
    a.add_argument("username")
    a.add_argument("--role", required=True, choices=ROLES)
    a.add_argument("--password-stdin", action="store_true")
    s = sub.add_parser("set-password")
    s.add_argument("username")
    s.add_argument("--password-stdin", action="store_true")
    r = sub.add_parser("set-role")
    r.add_argument("username")
    r.add_argument("role", choices=ROLES)
    for name in ("disable", "enable", "unlock"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    args = ap.parse_args(argv)

    cfg = AppConfig()
    svc = build_service(cfg)
    try:
        if args.cmd == "add-user":
            svc.add_user(args.username, args.role, _read_password(args.password_stdin))
            print(f"user {args.username.lower()} created ({args.role})")
        elif args.cmd == "set-password":
            ended = svc.set_password(args.username, _read_password(args.password_stdin))
            print(f"password changed; {ended} session(s) ended")
        elif args.cmd == "set-role":
            svc.set_role(args.username, args.role)
            print(f"role changed to {args.role}; sessions ended")
        elif args.cmd in ("disable", "enable"):
            svc.set_disabled(args.username, args.cmd == "disable")
            print(f"user {args.username.lower()} {args.cmd}d")
        elif args.cmd == "unlock":
            svc.unlock(args.username)
            print(f"failed-login lock cleared for {args.username.lower()}")
        else:
            users = svc.store.list_users()
            print(f"{'username':24s} {'role':9s} {'state':9s} {'last login':20s} password changed")
            for u in users:
                print(f"{u['username']:24s} {u['role']:9s} {'disabled' if u['disabled'] else 'active':9s} "
                      f"{_fmt(u['last_login_at']):20s} {_fmt(u['password_changed_at'])}")
            if not users:
                print("(no users)")
    except (ValueError, PasswordPolicyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if cfg.auth_mode != "on":
        print("note: LM_AUTH_MODE is off - accounts are stored but login is not active yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
