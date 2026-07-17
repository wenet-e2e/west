# Copyright (c) 2026 Pengshen Zhang
"""SSL Context: 本地 WSS 证书与 SSL 上下文。

- ensure_self_signed_cert: 自动生成本地自签名证书
- create_ssl_context: 为 websockets.serve 创建 server-side SSLContext
- 支持 cryptography 优先，openssl 命令兜底
"""
import datetime
import ipaddress
import logging
import os
import ssl
import subprocess
from pathlib import Path
from typing import Optional

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError:
    x509 = None
    hashes = None
    serialization = None
    rsa = None
    NameOID = None


def _generate_self_signed_cert_with_cryptography(
    *,
    cert_file: str,
    key_file: str,
) -> None:
    if (x509 is None or hashes is None or serialization is None
            or rsa is None or NameOID is None):
        raise RuntimeError("cryptography is not installed")

    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    Path(key_file).write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    Path(cert_file).write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _generate_self_signed_cert_with_openssl(
    *,
    cert_file: str,
    key_file: str,
) -> None:
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", key_file, "-out", cert_file,
        "-sha256", "-days", "365",
        "-nodes", "-subj", "/CN=localhost"
    ], check=True)


def generate_self_signed_cert(
    *,
    cert_file: str,
    key_file: str,
    logger: logging.Logger,
) -> None:
    try:
        _generate_self_signed_cert_with_cryptography(
            cert_file=cert_file,
            key_file=key_file,
        )
    except RuntimeError as e:
        logger.warning(
            f"{e}; falling back to openssl command")
        _generate_self_signed_cert_with_openssl(
            cert_file=cert_file,
            key_file=key_file,
        )


def create_ssl_context(
    *,
    use_ssl: bool,
    logger: logging.Logger,
    cert_file: str = "cert.pem",
    key_file: str = "key.pem",
) -> Optional[ssl.SSLContext]:
    """按需创建 HTTPS/WSS SSLContext。"""
    if not use_ssl:
        return None

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        logger.info("Generating self-signed SSL certificate...")
        generate_self_signed_cert(
            cert_file=cert_file,
            key_file=key_file,
            logger=logger,
        )
        logger.info(f"Generated {cert_file} and {key_file}")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, keyfile=key_file)
    logger.info("SSL context loaded. Server will run on HTTPS/WSS.")
    return ssl_context
