# 参考：https://github.com/graphdeco-inria/gaussian-splatting/blob/main/graphdecoviewer/widgets/__init__.py


from typing import Optional
from ..types import ViewerMode
from abc import ABC, abstractmethod

class Widget(ABC):
    id = 0

    def __init__(self, mode: ViewerMode):
        # 每个 widget 分配唯一 ID
        self.mode = mode
        self.widget_id = Widget.id
        Widget.id += 1

    def setup(self):
        """
        Perform any setup actions required after OpenGL/GLFW/ImGUI is initialized. 
        This function won't be called when application is running in headless mode.
        """
        # 子类可覆盖此函数

    def destroy(self):
        """
        Destroy any resources created manually in the 'setup' function.
        This function won't be called when application is running in headless mode.
        """
        # 子类可覆盖此函数

    def server_send(self) -> tuple[Optional[bytes], Optional[dict]]:
        """
        Send widget state to the client.

        Returns:
            binary (bytes): Any binary data to be sent to the client.
            text (dict): Any text data to be sent to the client.
        """
        # 默认无数据发送
        return None, None
    
    def server_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """
        Receive widget state from the client and update it.

        Args:
            binary (bytes): Any binary data received from the client.
            text (dict): Any text data sent received from the client
        """
        # 子类可覆盖此函数
    
    def client_send(self) -> tuple[Optional[bytes], Optional[dict]]:
        """
        Send widget state to the server.

        Returns:
            binary (bytes): Any binary data to be sent to the server.
            text (dict): Any text data to be sent to the server.
        """
        # 默认无数据发送
        return None, None

    def client_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """
        Send widget state to the server.

        Args:
            binary (bytes): Any binary data to be received from the server.
            text (dict): Any text data to be received from the server.
        """
        # 子类可覆盖此函数
    
    @abstractmethod
    def show_gui(self):
        pass