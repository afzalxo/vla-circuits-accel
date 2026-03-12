import socket
import struct
import numpy as np

class FPGAAgentClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to N-ISA FPGA Server at {host}:{port}")

    def get_action(self, image_np, command_id, prev_features, prev_actions, terminate):
        """
        image_np: [256, 256, 3] uint8 array
        command_id: int
        prev_actions: [steer, accel, speed] floats
        """
        steer, accel = None, None
        # 'i' = int32, 'f' = float32
        metadata = struct.pack('ii', 
                               command_id, 
                               terminate)
        self.sock.sendall(metadata)

        if not terminate:
            # Ensure it's in bytes/uint8 format
            self.sock.sendall(image_np.tobytes())
            self.sock.sendall(prev_features.tobytes())

            # 3. Receive Result
            # Wait for 8 bytes (2 floats)
            response_data = self.sock.recv(8)
            steer, accel = struct.unpack('ff', response_data)
        
        return steer, accel

    def close(self):
        self.sock.close()

# --- USAGE IN CARLA LOOP ---
# agent = FPGAAgentClient('FPGA_SERVER_IP', 3000)
# steer, accel = agent.get_action(camera_frame, current_cmd, last_act)
