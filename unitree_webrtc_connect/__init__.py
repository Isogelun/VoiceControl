# Monkey-patch aioice.Connection to use a fixed username and password accross all instances.

import asyncio
import logging
import os
import socket
import aioice


_original_get_host_addresses = aioice.ice.get_host_addresses
_original_stun_protocol_close = aioice.ice.StunProtocol.close
_original_connection_check_incoming = aioice.ice.Connection.check_incoming


def _infer_local_ip_for_peer(peer_ip: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((peer_ip, 9991))
        return sock.getsockname()[0]


def _filtered_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
    addresses = _original_get_host_addresses(use_ipv4, use_ipv6)
    configured = os.environ.get("UNITREE_WEBRTC_LOCAL_IP", "").strip()
    if not configured and os.environ.get("UNITREE_WEBRTC_AUTO_LOCAL_IP", "0").lower() in {"1", "true", "yes"}:
        peer_ip = os.environ.get("UNITREE_ROBOT_IP", "").strip()
        if peer_ip:
            try:
                configured = _infer_local_ip_for_peer(peer_ip)
            except OSError:
                configured = ""
    if configured:
        allowed = {item.strip() for item in configured.split(",") if item.strip()}
        filtered = [address for address in addresses if address in allowed]
        if filtered:
            return filtered
    return addresses


aioice.ice.get_host_addresses = _filtered_host_addresses  # type: ignore[attr-defined]


async def _safe_stun_protocol_close(self):
    timeout = float(os.environ.get("UNITREE_AIOICE_CLOSE_TIMEOUT", "1"))
    transport = getattr(self, "transport", None)
    closed = getattr(self, "_StunProtocol__closed", None)
    if transport is not None:
        transport.close()
    if closed is None or closed.done():
        return
    try:
        await asyncio.wait_for(asyncio.shield(closed), timeout=timeout)
    except asyncio.TimeoutError:
        logging.getLogger(__name__).debug("aioice StunProtocol.close timed out; forcing closed")
        if not closed.done():
            closed.set_result(True)


aioice.ice.StunProtocol.close = _safe_stun_protocol_close  # type: ignore[method-assign]


def _filtered_check_incoming(self, message, addr, protocol):
    allowed = os.environ.get("UNITREE_ROBOT_IP", "").strip()
    if os.environ.get("UNITREE_FILTER_REMOTE_CANDIDATES", "1").lower() not in {"0", "false", "no"}:
        if allowed and addr and addr[0] != allowed:
            logging.getLogger(__name__).debug("Ignoring peer reflexive candidate from %s; allowed=%s", addr[0], allowed)
            return
    return _original_connection_check_incoming(self, message, addr, protocol)


aioice.ice.Connection.check_incoming = _filtered_check_incoming  # type: ignore[method-assign]


class Connection(aioice.Connection):
    local_username = aioice.utils.random_string(4)
    local_password = aioice.utils.random_string(22)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_username = Connection.local_username
        self.local_password = Connection.local_password


aioice.Connection = Connection  # type: ignore


# Monkey-patch aiortc.rtcdtlstransport.X509_DIGEST_ALGORITHMS to remove extra SHA algorithms
# Extra SHA algorithms introduced in aiortc 1.10.0 causes Unity Go2 to use the new SCTP format, despite aiortc using the old SCTP syntax.
# This new format is not supported by aiortc version as of today (2025-06-02)


import aiortc
from packaging.version import Version


_original_pc_connect = aiortc.rtcpeerconnection.RTCPeerConnection._RTCPeerConnection__connect


async def _go2_safe_pc_connect(self):
    """Avoid an aiortc background-connect race seen with Go2 LAN answers.

    aiortc schedules __connect() once after setLocalDescription() and again
    after setRemoteDescription(). The first task can wake up before a remote
    description is available, then __remoteRtp() raises
    "'NoneType' object has no attribute 'media'" and the peer never finishes
    setting up SCTP/audio. Waiting briefly lets the second phase populate the
    description; returning still leaves the later scheduled connect task to run.
    """
    for _ in range(100):
        pending = getattr(self, "_RTCPeerConnection__pendingRemoteDescription", None)
        current = getattr(self, "_RTCPeerConnection__currentRemoteDescription", None)
        if pending is not None or current is not None:
            break
        await asyncio.sleep(0.01)
    else:
        logging.getLogger(__name__).debug("aiortc connect deferred until remote description is set")
        return

    for attempt in range(3):
        try:
            return await _original_pc_connect(self)
        except AttributeError as exc:
            if "'NoneType' object has no attribute 'media'" not in str(exc) or attempt == 2:
                raise
            await asyncio.sleep(0.05)


aiortc.rtcpeerconnection.RTCPeerConnection._RTCPeerConnection__connect = _go2_safe_pc_connect


if Version(aiortc.__version__) == Version("1.10.0"):
    X509_DIGEST_ALGORITHMS = {
        "sha-256": "SHA256",
    }
    aiortc.rtcdtlstransport.X509_DIGEST_ALGORITHMS = X509_DIGEST_ALGORITHMS

elif Version(aiortc.__version__) >= Version("1.11.0"):
    # Syntax changed in aiortc 1.11.0, so we need to use the hashes module
    from cryptography.hazmat.primitives import hashes

    X509_DIGEST_ALGORITHMS = {
        "sha-256": hashes.SHA256(),  # type: ignore
    }
    aiortc.rtcdtlstransport.X509_DIGEST_ALGORITHMS = X509_DIGEST_ALGORITHMS


# Public API
from .webrtc_driver import UnitreeWebRTCConnection  # noqa: E402
from .webrtc_datachannel import WebRTCDataChannel  # noqa: E402
from .constants import (  # noqa: E402
    WebRTCConnectionMethod,
    DATA_CHANNEL_TYPE,
    RTC_TOPIC,
    SPORT_CMD,
    SPORT_CMD_MCF,
    OBSTACLES_AVOID_API,
)
from .msgs.pub_sub import WebRTCDataChannelPubSub  # noqa: E402
from .unitree_cloud import (  # noqa: E402
    UnitreeCloud,
    UnitreeCloudError,
    RobotDevice,
    fetch_aes_key,
)
from .unitree_auth import (  # noqa: E402
    AesKeyRequiredError,
    AesKeyRejectedError,
    DataChannelTimeoutError,
    LocalSignalingPortError,
    NoSdpAnswerError,
    RobotBusyError,
)

__all__ = [
    "UnitreeWebRTCConnection",
    "WebRTCConnectionMethod",
    "WebRTCDataChannel",
    "WebRTCDataChannelPubSub",
    "DATA_CHANNEL_TYPE",
    "RTC_TOPIC",
    "SPORT_CMD",
    "SPORT_CMD_MCF",
    "OBSTACLES_AVOID_API",
    "UnitreeCloud",
    "UnitreeCloudError",
    "RobotDevice",
    "fetch_aes_key",
    "AesKeyRequiredError",
    "AesKeyRejectedError",
    "DataChannelTimeoutError",
    "LocalSignalingPortError",
    "NoSdpAnswerError",
    "RobotBusyError",
]
