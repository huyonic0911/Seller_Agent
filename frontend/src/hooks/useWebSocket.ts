import { useEffect, useRef, useState } from "react";

// Sự kiện đẩy từ backend qua /ws/stream
export type StreamMessage =
  | { type: "comment"; author: string; text: string; ts?: string }
  | {
      type: "reply";
      author: string;
      comment: string;
      text: string;
      audio: string;
      audio_format: string;
    }
  | { type: "error"; author?: string; comment?: string; message: string };

type Status = "connecting" | "open" | "closed";

/** Kết nối 1 WebSocket, tự reconnect, gọi onMessage cho từng message JSON. */
export function useStreamSocket(url: string, onMessage: (m: StreamMessage) => void) {
  const [status, setStatus] = useState<Status>("connecting");
  const cbRef = useRef(onMessage);
  cbRef.current = onMessage;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;
    let closed = false;

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(url);
      ws.onopen = () => setStatus("open");
      ws.onmessage = (ev) => {
        try {
          cbRef.current(JSON.parse(ev.data));
        } catch {
          /* bỏ qua message không phải JSON */
        }
      };
      ws.onclose = () => {
        setStatus("closed");
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws?.close();
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      ws?.close();
    };
  }, [url]);

  return status;
}

/** WebSocket gửi comment lên backend (/ws/comments). */
export function useCommentSender(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let retry: ReturnType<typeof setTimeout>;
    let closed = false;
    const connect = () => {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setReady(true);
      ws.onclose = () => {
        setReady(false);
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [url]);

  const send = (text: string, author = "khách") => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ text, author }));
    }
  };

  return { send, ready };
}
