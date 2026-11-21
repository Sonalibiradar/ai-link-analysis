"""URL validation with SSRF protection.

Rules:
  * only http/https, no embedded credentials, sane length and port
  * the hostname must resolve, and *every* resolved address must be globally routable
    (blocks loopback, RFC1918, link-local/cloud metadata 169.254.x.x, CGNAT, ULA, ...)
  * numeric/obfuscated hosts (2130706433, 0x7f.1, ...) are handled because we validate
    the addresses the resolver actually returns, not the text of the hostname.

Known limitation: a DNS-rebinding attacker could change the record between this check
and the connection. For production, pin the connection to the validated IP (or run the
fetcher in an egress-restricted network). This is documented in the README.
"""

import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

from app.errors import BlockedURLError, InvalidURLError

ALLOWED_PORTS = {80, 443, 8080, 8443}
MAX_URL_LENGTH = 2048
BLOCKED_HOST_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet")


def normalize_url(raw: str) -> str:
    """Trim, add https:// when the scheme is missing, and return a canonical string."""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidURLError("A website URL is required.")
    raw = raw.strip()
    if len(raw) > MAX_URL_LENGTH:
        raise InvalidURLError("URL is too long.")
    if any(c.isspace() for c in raw) or any(ord(c) < 32 for c in raw):
        raise InvalidURLError("URL must not contain whitespace or control characters.")
    if "://" not in raw:
        raw = "https://" + raw
    return raw


def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def resolve_host(host: str, port: int) -> list[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise InvalidURLError(f"Could not resolve host '{host}'. Check the domain name.")
    except UnicodeError:
        raise InvalidURLError("Invalid hostname.")
    ips = []
    for info in infos:
        try:
            ips.append(ipaddress.ip_address(info[4][0].split("%")[0]))
        except ValueError:
            continue
    if not ips:
        raise InvalidURLError(f"Could not resolve host '{host}'.")
    return ips


def validate_public_url(raw: str) -> str:
    """Return a normalized URL, or raise InvalidURLError / BlockedURLError."""
    url = normalize_url(raw)
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise InvalidURLError("URL is malformed.")

    if parts.scheme.lower() not in ("http", "https"):
        raise InvalidURLError("Only http:// and https:// URLs are supported.")
    if parts.username or parts.password:
        raise BlockedURLError("URLs with embedded credentials are not allowed.")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise InvalidURLError("URL has no hostname.")
    if port is not None and port not in ALLOWED_PORTS:
        raise BlockedURLError(f"Port {port} is not allowed.")
    if host == "localhost" or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise BlockedURLError("Internal hostnames are not allowed.")

    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise InvalidURLError("Invalid hostname.")

    is_ip_literal = False
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip_literal = True
    except ValueError:
        pass
    # A bare TLD-less name ("intranet", "router") is almost certainly internal.
    if not is_ip_literal and "." not in ascii_host:
        raise InvalidURLError("Please provide a full domain name, e.g. example.com.")

    resolve_port = port or (443 if parts.scheme.lower() == "https" else 80)
    for ip in resolve_host(ascii_host, resolve_port):
        if not _is_public_ip(ip):
            raise BlockedURLError("This address points to a private or internal network and cannot be analyzed.")

    netloc = ascii_host if port is None else f"{ascii_host}:{port}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))
