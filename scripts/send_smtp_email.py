#!/usr/bin/env python3
"""
Send email with attachment using SMTP (stdlib only — no Outlook).

Credentials and server come from environment variables only (never commit values).

Gmail example (2FA on account → Google Account → App passwords):
  set FINETUNER_SMTP_HOST=smtp.gmail.com
  set FINETUNER_SMTP_PORT=587
  set FINETUNER_SMTP_USER=you@gmail.com
  set FINETUNER_SMTP_PASSWORD=your_16_char_app_password
  set FINETUNER_MAIL_FROM=you@gmail.com

Then:
  python scripts/send_smtp_email.py --to someone@gmail.com --attach path\\file.docx

Other providers: set host/port; use FINETUNER_SMTP_SSL=1 for implicit SSL (e.g. port 465).
"""
from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is not None and v.strip() != "":
        return v
    return default


def send_with_attachment(
    *,
    to_addrs: list[str],
    subject: str,
    body: str,
    attachment_path: Path,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    mail_from: str,
    use_ssl: bool,
) -> None:
    if not attachment_path.is_file():
        raise FileNotFoundError(f"Attachment not found: {attachment_path}")

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with attachment_path.open("rb") as f:
        part = MIMEBase("application", "vnd.openxmlformats-officedocument.wordprocessingml.document")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename=attachment_path.name)
    msg.attach(part)

    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(mail_from, to_addrs, msg.as_string())
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_user, smtp_password)
            server.sendmail(mail_from, to_addrs, msg.as_string())


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    default_attach = repo / "docs" / "ARTICLE_SME_REVIEW_EDITABLE.docx"

    p = argparse.ArgumentParser(description="Send SMTP email with attachment (env-based auth).")
    p.add_argument("--to", required=True, help="Recipient email (comma-separated for multiple).")
    p.add_argument("--subject", default="SME review: ARTICLE_SME_REVIEW_EDITABLE.docx")
    p.add_argument("--body", default="Attached: editable Word document for SME review.")
    p.add_argument("--attach", type=Path, default=default_attach, help="File to attach.")
    args = p.parse_args()

    host = _env("FINETUNER_SMTP_HOST", "smtp.gmail.com")
    port_s = _env("FINETUNER_SMTP_PORT", "587")
    user = _env("FINETUNER_SMTP_USER")
    password = _env("FINETUNER_SMTP_PASSWORD")
    mail_from = _env("FINETUNER_MAIL_FROM") or user
    use_ssl = (_env("FINETUNER_SMTP_SSL", "0") or "0").strip().lower() in ("1", "true", "yes")

    if not host or not user or not password or not mail_from:
        print(
            "Missing SMTP configuration. Set these environment variables:\n"
            "  FINETUNER_SMTP_HOST (default: smtp.gmail.com)\n"
            "  FINETUNER_SMTP_PORT (default: 587)\n"
            "  FINETUNER_SMTP_USER\n"
            "  FINETUNER_SMTP_PASSWORD\n"
            "  FINETUNER_MAIL_FROM (optional; defaults to FINETUNER_SMTP_USER)\n"
            "  FINETUNER_SMTP_SSL=1 for port 465 implicit SSL\n",
            file=sys.stderr,
        )
        return 2

    try:
        port = int(port_s or "587")
    except ValueError:
        print(f"Invalid FINETUNER_SMTP_PORT: {port_s!r}", file=sys.stderr)
        return 2

    to_list = [x.strip() for x in args.to.split(",") if x.strip()]
    send_with_attachment(
        to_addrs=to_list,
        subject=args.subject,
        body=args.body,
        attachment_path=args.attach.resolve(),
        smtp_host=host,
        smtp_port=port,
        smtp_user=user,
        smtp_password=password,
        mail_from=mail_from,
        use_ssl=use_ssl,
    )
    print(f"Sent to {', '.join(to_list)} with attachment {args.attach.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
