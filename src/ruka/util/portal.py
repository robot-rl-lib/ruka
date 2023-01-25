import fcntl
import os
import pickle
import select
import socket
import struct

from typing import Any, Callable
from ruka_os.globals import PORTAL_SOCK_TPL


def get_portal_file(name:str):
    return PORTAL_SOCK_TPL.replace('%name%', name)


def set_fd_flag(fd, flag):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL, 0)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | flag)


def open_portal(name: str, non_block: bool=False):
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(get_portal_file(name))
        if non_block:
            set_fd_flag(sock, os.O_NONBLOCK) 
        return Portal(sock)
    except:
        return None


class Portal:
    """
    Handles the communication on a connected socket
    """
    def __init__(self, sock):
        self.sock = sock

    def __del__(self):
        self.sock.close()
        
    def send(self, data: Any):
        """
        Sends Any data to pipe, previously pickle it and addthe header containing
        size of the packet
        """
        b = pickle.dumps(data)
        size = len(b)
        s = ''+chr(0)
        s = str.encode(s)
        header = bytearray(struct.pack('<cL', s, size))
        r = self.sock.send(header)
        if r != len(header):
            return False
        r = self.sock.send(b)
        if r != size:
            return False
        return True

    def recv_control(self):
        h1 = self.sock.recv(1)
        if len(h1) == 0:
            raise ConnectionAbortedError()
        return h1

    def recv(self) -> Any:
        """
        Reads header and arriving size of hte packet, then unpickles it to Any object
        """
        h1 = self.recv_control()
        if h1 != b'\0':
            raise RuntimeError('Bad packet on socket')        
        h2 = self.sock.recv(4)
        size = struct.unpack("<L", h2)
        size =size[0]
        data = self.sock.recv(size)
        return pickle.loads(data)
        
    def send_char(self, b: str):
        """
        Special function to send only 1 byte, without picling and etc

        b - should not be the zero \0 symbol
        """
        header = bytearray(struct.pack('<c',b[0]))
        r = self.sock.send(header)
        return r==1

    def recv_char(self):
        """
        Special function to read only 1 byte, without picling and etc

        Returns 0 is failed
        """
        h1 = self.recv_control()
        if h1 == b'\0':
            raise RuntimeError('Bad packet on pipe')
        return h1

    def wait_for_data(self):
        fds = [self.sock]
        reads, _, _ = select.select(fds, [], [])
        for cn in reads:
            if cn==self.sock:
                return True
        return False


class PortalSystemLinear:
    """
    Creates server socket and waits for connections

    Doesn't lights up any thread or process, everything is done in main loop
    """
    def __init__(self, name, portal_procesor: Callable):
        self._portal_processor = portal_procesor
        self._name = name
        self._sock_file = get_portal_file(self._name)
        if os.path.exists(self._sock_file):
            os.remove(self._sock_file)

        self.serv_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.serv_socket.bind(self._sock_file)
        self.serv_socket.listen(10)
        os.chmod(self._sock_file, 0o777)
        
        self._portals = {}


    def __del__(self):
        self.serv_socket.close()
        os.remove(self._sock_file)

    def loop(self):
        while True:
            fds = []
            fds.append(self.serv_socket)
            fds.extend(self._portals.keys())
            reads, _, errs = select.select(fds, [], fds)
            # process exceptions
            for cn in errs:
                # server socket is dead, how?!
                if cn==self.serv_socket:
                    return False

                # remove an open portal
                try:
                    del self._portals[cn]
                except:
                    pass

            # process read
            for cn in reads:

                # new client arrived
                if cn==self.serv_socket:
                    (client, address) = self.serv_socket.accept()
                    self._portals[client] = Portal(client)

                # process client
                else:                    
                    ret = True
                    try:
                        ret = self._portal_processor(self._portals[cn])
                    except ConnectionAbortedError:
                        ret = False

                    # drop portal
                    if not ret:
                        del self._portals[cn]

        return False


