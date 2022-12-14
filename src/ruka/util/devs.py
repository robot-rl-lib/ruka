import evdev


def get_avail_virtual_kbd_devname() -> str:
    num = 0
    name = 'teleop-kbd'
    pref = name
    while True:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        devices = [d for d in devices if d.name == name]
        if not devices:
            break 
        num += 1
        name = f'{pref}-{num}'
    return name