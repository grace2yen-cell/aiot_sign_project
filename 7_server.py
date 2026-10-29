"""
7_server.py
[실제 사용] 수화 인식 서버 - 실행 진입점
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 클라이언트로부터 TCP로 영상 파일을 받아 저장
2. server_support_with_postprocess.main()으로 수화 인식
3. 인식 결과(클래스 이름)를 UDP로 클라이언트에 전송
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import socket
import struct

from server_support_with_postprocess import main as run_inference
from config import TCP_HOST, TCP_PORT, UDP_PORT, SAVE_NAME


def receive_mp4(conn, save_path):
    raw = conn.recv(8)
    filesize = struct.unpack(">Q", raw)[0]

    with open(save_path, "wb") as f:
        received = 0
        while received < filesize:
            chunk = conn.recv(4096)
            if not chunk:
                break
            f.write(chunk)
            received += len(chunk)


def udp_send(ip, msg):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.sendto(msg.encode(), (ip, UDP_PORT))


def server_loop():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)

    while True:
        print("[SERVER] 대기 중…")
        conn, addr = server.accept()
        client_ip = addr[0]
        print(f"[TCP] 연결됨: {client_ip}")

        with conn:
            receive_mp4(conn, SAVE_NAME)

            print("[INFO] 추론 시작")
            result = run_inference(SAVE_NAME, show_video=True)

            print(f"[INFO] 추론 결과: {result}")
            udp_send(client_ip, result)


if __name__ == "__main__":
    server_loop()
