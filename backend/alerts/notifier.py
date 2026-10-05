"""Notification providers (email, webhook, Telegram). All disabled unless their
credentials exist in the environment. Sending happens on a background thread so
a slow SMTP server never blocks detection.

Per-camera control: ``cameras.notify_json`` = {"enabled": true, "min_severity": "MEDIUM",
"providers": ["email", "telegram"]}. Global default comes from NOTIFY_MIN_SEVERITY.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import smtplib
import threading
import urllib.request
from email.message import EmailMessage
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _text(alert: dict[str, Any]) -> str:
    return (f"[{alert['severity']}] Litter detected: {alert['class']} ({alert['confidence']:.0%}) "
            f"camera {alert['camera_id']} track {alert.get('track_id')} at {alert['ts']} "
            f"(alert #{alert['id']})")


class Provider:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def send(self, alert: dict[str, Any]) -> None:
        raise NotImplementedError


class WebhookProvider(Provider):
    name = "webhook"

    def available(self) -> bool:
        return bool(os.getenv("ALERT_WEBHOOK_URL"))

    def send(self, alert: dict[str, Any]) -> None:
        url = os.environ["ALERT_WEBHOOK_URL"]
        if not url.startswith(("https://", "http://")):
            raise ValueError("ALERT_WEBHOOK_URL must be http(s)")
        body = json.dumps({"text": _text(alert), "alert": alert}, default=str).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).close()  # noqa: S310 - scheme validated above


class TelegramProvider(Provider):
    name = "telegram"

    def available(self) -> bool:
        return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))

    def send(self, alert: dict[str, Any]) -> None:
        url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
        body = json.dumps({"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": _text(alert)}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).close()  # noqa: S310


class EmailProvider(Provider):
    name = "email"

    def available(self) -> bool:
        return all(os.getenv(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"))

    def send(self, alert: dict[str, Any]) -> None:
        msg = EmailMessage()
        msg["Subject"] = f"[{alert['severity']}] Litter alert #{alert['id']} - {alert['camera_id']}"
        msg["From"] = os.getenv("ALERT_EMAIL_FROM", os.environ["SMTP_USER"])
        msg["To"] = os.environ["ALERT_EMAIL_TO"]
        msg.set_content(_text(alert))
        crop = alert.get("crop_path")
        if crop and os.path.isfile(crop):
            with open(crop, "rb") as fh:
                msg.add_attachment(fh.read(), maintype="image", subtype="jpeg", filename=os.path.basename(crop))
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587")), timeout=15) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)


class Notifier:
    def __init__(self, camera_prefs: Callable[[str], dict[str, Any]],
                 providers: Optional[list[Provider]] = None,
                 on_error: Optional[Callable[[str, str], None]] = None,
                 on_delivered: Optional[Callable[[str, dict[str, Any]], None]] = None) -> None:
        self.providers = providers if providers is not None else [EmailProvider(), WebhookProvider(),
                                                                  TelegramProvider()]
        self.camera_prefs = camera_prefs
        self.on_error = on_error
        self.on_delivered = on_delivered   # called ONLY after a provider call returned without error
        self.default_min = os.getenv("NOTIFY_MIN_SEVERITY", "HIGH").upper()
        self._q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=500)
        threading.Thread(target=self._worker, name="notifier", daemon=True).start()

    def status(self) -> list[dict[str, Any]]:
        return [{"provider": p.name, "configured": p.available()} for p in self.providers]

    def targets(self, alert: dict[str, Any]) -> list[Provider]:
        prefs = self.camera_prefs(alert["camera_id"]) or {}
        if not prefs.get("enabled", True):
            return []
        min_sev = str(prefs.get("min_severity", self.default_min)).upper()
        if SEVERITY_ORDER.get(alert["severity"], 0) < SEVERITY_ORDER.get(min_sev, 2):
            return []
        allowed = prefs.get("providers")
        return [p for p in self.providers if p.available() and (not allowed or p.name in allowed)]

    def dispatch(self, alert: dict[str, Any]) -> None:
        if not self.targets(alert):
            return
        try:
            self._q.put_nowait(alert)
        except queue.Full:
            log.error("Notification queue full; dropping notification for alert %s", alert.get("id"))

    def _worker(self) -> None:
        while True:
            alert = self._q.get()
            for p in self.targets(alert):
                try:
                    p.send(alert)
                    if self.on_delivered:
                        self.on_delivered(p.name, alert)
                except Exception as exc:  # never crash the worker
                    log.error("Notification via %s failed: %s", p.name, exc)
                    if self.on_error:
                        self.on_error(p.name, str(exc))
