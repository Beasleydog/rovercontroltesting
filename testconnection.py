import socketio

sio = socketio.Client()

@sio.event
def connect():
    print("connected", sio.sid)

@sio.event
def disconnect():
    print("disconnected")

@sio.on("rover-telemetry")
def on_rover(data):
    print("rover", data)

@sio.on("ltv-telemetry")
def on_ltv(data):
    print("ltv", data)

sio.connect("http://35.3.249.68:5001")
sio.wait()