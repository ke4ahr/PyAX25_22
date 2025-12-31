# pyax25_22/interfaces/kiss_async.py
"""
Asynchronous KISS Interface Implementation

Implements non-blocking KISS communication with multi-drop support.

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import asyncio
import logging
from typing import Optional, Callable

from .kiss import (
    KISSInterface,
    KISSCommand,
    KISSProtocolError,
    TransportError
)

logger = logging.getLogger(__name__)

class AsyncKISSInterface(KISSInterface):
    """
    Asynchronous KISS interface using asyncio.
    
    Args:
        tnc_address: TNC address for multi-drop (0-15)
        poll_interval: Poll interval in seconds
        max_queue_size: Maximum receive queue size
    """
    def __init__(
        self,
        tnc_address: int = 0,
        poll_interval: float = 0.1,
        max_queue_size: int = 100
    ):
        super().__init__(tnc_address=tnc_address, poll_interval=poll_interval)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._receive_queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self, host: str, port: int) -> None:
        """
        Connect to a KISS TCP server.
        
        Args:
            host: Server hostname/IP
            port: Server port
        """
        if self._running:
            return
            
        try:
            self._reader, self._writer = await asyncio.open_connection(host, port)
            self._running = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info(f"Connected to {host}:{port}")
        except OSError as e:
            raise TransportError(f"Connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close the connection"""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        logger.info("Disconnected")

    async def send_frame(
        self,
        data: bytes,
        cmd: int = KISSCommand.DATA
    ) -> None:
        """
        Asynchronously send a KISS frame.
        
        Args:
            data: Frame payload
            cmd: KISS command (default DATA)
        """
        if not self._writer:
            raise TransportError("Not connected")
            
        cmd_byte = self._encode_command(cmd)
        frame = self._build_frame(cmd_byte, data)
        
        try:
            self._writer.write(frame)
            await self._writer.drain()
            logger.debug(f"Sent frame (cmd=0x{cmd:02x}, len={len(data)})")
        except OSError as e:
            raise TransportError(f"Send failed: {e}") from e

    async def send_poll(self, target_tnc: int) -> None:
        """
        Send poll command to target TNC.
        
        Args:
            target_tnc: TNC address (0-15)
        """
        cmd_byte = (target_tnc << 4) | KISSCommand.POLL
        await self.send_frame(b'', cmd=cmd_byte)

    async def recv_frame(
        self,
        timeout: Optional[float] = None
    ) -> Optional[Tuple[bytes, int]]:
        """
        Receive a frame asynchronously.
        
        Args:
            timeout: Maximum wait time (seconds)
        
        Returns:
            Tuple of (frame, tnc_address) or None on timeout
        """
        try:
            return await asyncio.wait_for(
                self._receive_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def _receive_loop(self) -> None:
        """Main receive loop"""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running and self._reader:
            try:
                data = await self._reader.read(1024)
                if not data:
                    break  # Connection closed
                    
                for byte in data:
                    if byte == self.FEND:
                        if in_frame and buffer:
                            await self._process_frame(bytes(buffer))
                            buffer.clear()
                        in_frame = True
                        escaped = False
                    elif in_frame:
                        if escaped:
                            if byte == self.TFEND:
                                buffer.append(self.FEND)
                            elif byte == self.TFESC:
                                buffer.append(self.FESC)
                            else:
                                logger.warning(f"Invalid escape byte: 0x{byte:02x}")
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                        else:
                            buffer.append(byte)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break
                
        self._running = False
        if self._writer:
            self._writer.close()

    async def _process_frame(self, frame: bytes) -> None:
        """Handle complete received frame"""
        if not frame:
            return
            
        cmd_byte = frame[0]
        tnc_address = (cmd_byte >> 4) & 0x0F
        cmd = cmd_byte & 0x0F
        payload = frame[1:]
        
        try:
            if cmd == KISSCommand.POLL:
                if self._poll_callback:
                    await self._poll_callback(tnc_address)
            else:
                await self._receive_queue.put((payload, tnc_address))
                if self._rx_callback:
                    await self._rx_callback(payload, tnc_address)
        except Exception as e:
            logger.error(f"Frame handler error: {e}")

    def __repr__(self) -> str:
        return (f"AsyncKISSInterface(tnc={self.tnc_address}, "
                f"connected={self._running})")

# Example usage
async def main():
    kiss = AsyncKISSInterface(tnc_address=1)
    
    def frame_handler(frame: bytes, tnc: int) -> None:
        print(f"Frame from TNC {tnc}: {frame.hex()}")
    
    kiss.register_rx_callback(frame_handler)
    
    try:
        await kiss.connect("localhost", 8001)
        
        # Send poll every 5 seconds
        while True:
            await kiss.send_poll(2)
            frame = await kiss.recv_frame(timeout=5)
            if frame:
                print(f"Received: {frame[0].hex()}")
            await asyncio.sleep(5)
    finally:
        await kiss.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
