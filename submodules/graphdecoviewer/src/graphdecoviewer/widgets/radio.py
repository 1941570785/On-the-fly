# 参考：https://github.com/graphdeco-inria/gaussian-splatting/blob/main/graphdecoviewer/widgets/radio.py


import string
import random
from enum import Enum
from imgui_bundle import imgui
from ..widgets import Widget
from ..types import ViewerMode

class RadioPicker(Widget):
    def __init__(self, mode: ViewerMode, default: Enum):
        # 当前选中值与状态表
        self.value = default
        self.states = dict.fromkeys(type(default), False)
        self.states[default] = True
        # Generate a random suffix to avoid collisions.
        # Technically a collision is still possible but highly unlikely.
        # 用随机后缀避免 ImGui ID 冲突
        self.rand = "##" + "".join(random.choices(string.ascii_letters + string.digits, k=8))
        super().__init__(mode)

    def show_gui(self) -> bool:
        for option, _ in self.states.items():
            # 渲染单选按钮并同步状态
            if imgui.radio_button(option.name.capitalize() + self.rand, self.states[option]):
                if option != self.value:
                    self.states[option] = True
                    self.states[self.value] = False
                    self.value = option

                    return True

        return False
    
    def client_send(self):
        # 将当前选中值发送到服务端
        return None, {"value": self.value}
    
    def server_recv(self, _, text):
        # 服务端更新选中值
        self.value = type(self.value)(text["value"])