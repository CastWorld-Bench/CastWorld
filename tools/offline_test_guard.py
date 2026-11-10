"""Pytest plugin preventing accidental external connections during release tests."""
import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    original = socket.socket.connect

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0]
            try:
                allowed = ipaddress.ip_address(host).is_loopback
            except ValueError:
                allowed = host == 'localhost'
            if not allowed:
                raise RuntimeError('External network disabled during release tests')
        return original(sock, address)

    monkeypatch.setattr(socket.socket, 'connect', connect)
