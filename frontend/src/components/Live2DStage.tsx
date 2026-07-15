import { useEffect, useRef, useState } from "react";
import { config } from "../config";
import { AudioLipSync } from "../lib/lipsync";
import { initLive2D, Live2DController } from "../lib/live2d";

interface Props {
  lipSync: AudioLipSync;
  speaking: boolean;
}

/**
 * Sân khấu avatar. Nếu VITE_USE_LIVE2D=true và nạp model thành công → render Live2D;
 * ngược lại render avatar SVG (mặc định, luôn chạy được) — cả hai đều nhép miệng
 * theo biên độ audio do AudioLipSync cung cấp.
 */
export function Live2DStage({ lipSync, speaking }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouthRef = useRef<SVGGElement>(null);
  const [useLive2D, setUseLive2D] = useState(config.useLive2D);

  // Nạp Live2D (nếu bật). Lỗi → fallback SVG.
  useEffect(() => {
    if (!config.useLive2D || !canvasRef.current) return;
    let ctrl: Live2DController | null = null;
    let unsub: (() => void) | null = null;
    let alive = true;

    initLive2D(canvasRef.current, config.live2dModel)
      .then((c) => {
        if (!alive) {
          c.destroy();
          return;
        }
        ctrl = c;
        unsub = lipSync.onLevel((lvl) => ctrl?.setMouth(lvl));
      })
      .catch((err) => {
        console.warn("Không nạp được Live2D, dùng avatar SVG:", err);
        setUseLive2D(false);
      });

    return () => {
      alive = false;
      unsub?.();
      ctrl?.destroy();
    };
  }, [lipSync]);

  // Avatar SVG: điều khiển độ mở miệng bằng scaleY của nhóm miệng (imperatively).
  useEffect(() => {
    if (useLive2D) return;
    const unsub = lipSync.onLevel((lvl) => {
      const g = mouthRef.current;
      if (g) g.style.transform = `translateY(${lvl * 6}px) scaleY(${0.15 + lvl * 1.6})`;
    });
    return unsub;
  }, [lipSync, useLive2D]);

  return (
    <div className="stage">
      {useLive2D ? (
        <canvas ref={canvasRef} className="live2d-canvas" />
      ) : (
        <SvgAvatar mouthRef={mouthRef} speaking={speaking} />
      )}
    </div>
  );
}

function SvgAvatar({
  mouthRef,
  speaking,
}: {
  mouthRef: React.RefObject<SVGGElement>;
  speaking: boolean;
}) {
  return (
    <svg viewBox="0 0 240 300" className={`svg-avatar ${speaking ? "speaking" : ""}`}>
      {/* tóc sau */}
      <ellipse cx="120" cy="130" rx="92" ry="105" fill="#3a2b45" />
      {/* mặt */}
      <ellipse cx="120" cy="140" rx="72" ry="82" fill="#ffe0c7" />
      {/* má hồng */}
      <circle cx="82" cy="160" r="13" fill="#ffb3a7" opacity="0.55" />
      <circle cx="158" cy="160" r="13" fill="#ffb3a7" opacity="0.55" />
      {/* mắt */}
      <g className="eyes">
        <ellipse cx="95" cy="132" rx="9" ry="12" fill="#2b2333" />
        <ellipse cx="145" cy="132" rx="9" ry="12" fill="#2b2333" />
        <circle cx="98" cy="128" r="3" fill="#fff" />
        <circle cx="148" cy="128" r="3" fill="#fff" />
      </g>
      {/* lông mày */}
      <path d="M82 112 q13 -8 26 0" stroke="#5a4763" strokeWidth="3" fill="none" strokeLinecap="round" />
      <path d="M132 112 q13 -8 26 0" stroke="#5a4763" strokeWidth="3" fill="none" strokeLinecap="round" />
      {/* miệng — nhóm này được scaleY theo biên độ audio */}
      <g transform="translate(120 178)">
        <g ref={mouthRef} style={{ transformOrigin: "center", transformBox: "fill-box" }}>
          <ellipse cx="0" cy="0" rx="18" ry="12" fill="#a83b4b" />
          <ellipse cx="0" cy="4" rx="12" ry="6" fill="#e8637a" />
        </g>
      </g>
      {/* tóc trước */}
      <path d="M48 120 q-6 -70 72 -76 q78 6 72 76 q-30 -34 -72 -30 q-42 -4 -72 30 z" fill="#4a3557" />
    </svg>
  );
}
