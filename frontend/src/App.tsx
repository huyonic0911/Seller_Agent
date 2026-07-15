import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AdminPanel } from "./components/AdminPanel";
import { ChatItem, ChatOverlay } from "./components/ChatOverlay";
import { Live2DStage } from "./components/Live2DStage";
import { config } from "./config";
import { useCommentSender, useStreamSocket, StreamMessage } from "./hooks/useWebSocket";
import { AudioLipSync } from "./lib/lipsync";

let uid = 0;

export default function App() {
  const lipSync = useMemo(() => new AudioLipSync(), []);
  const [items, setItems] = useState<ChatItem[]>([]);
  const [currentReply, setCurrentReply] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [audioReady, setAudioReady] = useState(false);

  // Hàng đợi phát audio: reply tới liên tục, phát tuần tự từng câu.
  const queueRef = useRef<{ text: string; audio: string }[]>([]);
  const playingRef = useRef(false);

  const pushItem = (it: Omit<ChatItem, "id">) =>
    setItems((prev) => [...prev.slice(-30), { ...it, id: ++uid }]);

  const drain = useCallback(async () => {
    if (playingRef.current) return;
    playingRef.current = true;
    while (queueRef.current.length) {
      const next = queueRef.current.shift()!;
      setCurrentReply(next.text);
      setSpeaking(true);
      try {
        if (next.audio) await lipSync.play(next.audio);
        else await new Promise((r) => setTimeout(r, 1200)); // không có audio → hiển thị 1 nhịp
      } catch (e) {
        console.warn("Lỗi phát audio:", e);
      }
      setSpeaking(false);
    }
    setCurrentReply(null);
    playingRef.current = false;
  }, [lipSync]);

  const onMessage = useCallback(
    (m: StreamMessage) => {
      if (m.type === "comment") {
        pushItem({ kind: "comment", author: m.author, text: m.text });
      } else if (m.type === "reply") {
        pushItem({ kind: "reply", author: m.author, comment: m.comment, text: m.text });
        queueRef.current.push({ text: m.text, audio: m.audio });
        void drain();
      } else if (m.type === "error") {
        pushItem({ kind: "error", text: m.message });
      }
    },
    [drain],
  );

  const status = useStreamSocket(config.wsStream, onMessage);
  const { send, ready } = useCommentSender(config.wsComments);

  const enableAudio = async () => {
    await lipSync.resume();
    setAudioReady(true);
  };

  useEffect(() => () => lipSync.stop(), [lipSync]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">🛍️ Seller Agent — Live 2D</div>
        <div className={`conn ${status}`}>Kết nối: {status}</div>
        {!audioReady && (
          <button className="enable-audio" onClick={enableAudio}>
            🔊 Bật âm thanh
          </button>
        )}
      </header>

      <main className="live-area">
        <Live2DStage lipSync={lipSync} speaking={speaking} />
        <ChatOverlay items={items} currentReply={currentReply} />
      </main>

      <footer className="footer">
        <AdminPanel onSend={send} ready={ready} />
        {!audioReady && (
          <p className="hint">Bấm "Bật âm thanh" trước để nghe agent đọc (chính sách autoplay của trình duyệt).</p>
        )}
      </footer>
    </div>
  );
}
