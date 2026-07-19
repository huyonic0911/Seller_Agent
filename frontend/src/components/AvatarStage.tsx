import { useEffect, useRef, useState } from "react";
import { config } from "../config";
import { AudioLipSync } from "../lib/lipsync";
import { ActionName, AvatarController, initVRM } from "../lib/vrm";

interface Props {
  lipSync: AudioLipSync;
  speaking: boolean;
  emotion: string;
}

// Danh sách nút action (nhãn tiếng Việt + icon). Thêm action mới: khai báo ở
// ActionName trong vrm.ts, cài logic trong vòng lặp, rồi thêm 1 dòng ở đây.
const ACTIONS: { name: ActionName; label: string; icon: string }[] = [
  { name: "closeEyes", label: "Nhắm mắt", icon: "😌" },
  { name: "raiseHand", label: "Giơ tay", icon: "✋" },
  { name: "wave", label: "Vẫy tay", icon: "👋" },
  { name: "clap", label: "Vỗ tay", icon: "👏" },
  { name: "nod", label: "Gật đầu", icon: "🙆" },
  { name: "shake", label: "Lắc đầu", icon: "🙅" },
];

/**
 * Sân khấu avatar VRM (three.js + three-vrm). Nhân vật nhép miệng theo biên độ audio
 * (AudioLipSync) và đổi biểu cảm theo `emotion` do backend gắn cho mỗi câu trả lời.
 * Không có fallback: nếu nạp model lỗi thì hiển thị thông báo lỗi.
 */
export function AvatarStage({ lipSync, speaking, emotion }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctrlRef = useRef<AvatarController | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  // Nạp VRM một lần.
  useEffect(() => {
    if (!canvasRef.current) return;
    let ctrl: AvatarController | null = null;
    let unsub: (() => void) | null = null;
    let alive = true;

    initVRM(canvasRef.current, config.vrmModel)
      .then((c) => {
        if (!alive) {
          c.destroy();
          return;
        }
        ctrl = c;
        ctrlRef.current = c;
        unsub = lipSync.onLevel((lvl) => c.setMouth(lvl));
        setStatus("ready");
      })
      .catch((err) => {
        console.error("Không nạp được model VRM:", err);
        setStatus("error");
      });

    return () => {
      alive = false;
      unsub?.();
      ctrl?.destroy();
      ctrlRef.current = null;
    };
  }, [lipSync]);

  // Đổi biểu cảm theo cảm xúc.
  useEffect(() => {
    ctrlRef.current?.setEmotion(emotion);
  }, [emotion]);

  return (
    <div className="stage">
      <canvas ref={canvasRef} className={`vrm-canvas ${speaking ? "speaking" : ""}`} />
      {status === "loading" && <div className="stage-overlay">Đang tải nhân vật…</div>}
      {status === "error" && (
        <div className="stage-overlay stage-error">
          Không tải được nhân vật VRM.
          <br />
          Kiểm tra file <code>{config.vrmModel}</code> trong <code>public/models/</code>.
        </div>
      )}
      {status === "ready" && (
        <>
          <div className="action-bar">
            {ACTIONS.map((a) => (
              <button
                key={a.name}
                className="action-btn"
                title={a.label}
                onClick={() => ctrlRef.current?.playAction(a.name)}
              >
                <span className="action-icon">{a.icon}</span>
                <span className="action-label">{a.label}</span>
              </button>
            ))}
          </div>
          <div className="zoom-bar">
            <button className="zoom-btn" title="Zoom in" onClick={() => ctrlRef.current?.zoom(-0.4)}>
              ＋
            </button>
            <button className="zoom-btn" title="Zoom out" onClick={() => ctrlRef.current?.zoom(0.4)}>
              －
            </button>
          </div>
        </>
      )}
    </div>
  );
}
