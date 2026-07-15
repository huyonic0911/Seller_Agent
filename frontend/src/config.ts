// Cấu hình frontend (đọc từ biến môi trường Vite, có mặc định hợp lý).

const BACKEND_HOST = import.meta.env.VITE_BACKEND_HOST ?? "localhost:8000";

export const config = {
  // WebSocket tới backend
  wsStream: `ws://${BACKEND_HOST}/ws/stream`,
  wsComments: `ws://${BACKEND_HOST}/ws/comments`,
  httpBase: `http://${BACKEND_HOST}`,

  // Live2D: bật khi đã đặt model vào public/models và có Cubism Core.
  // Mặc định false → dùng avatar SVG có nhép miệng, chạy được ngay.
  useLive2D: (import.meta.env.VITE_USE_LIVE2D ?? "false") === "true",
  // Đường dẫn model .model3.json (Cubism 4) hoặc .model.json (Cubism 2)
  live2dModel: import.meta.env.VITE_LIVE2D_MODEL ?? "/models/hiyori/hiyori_pro_t11.model3.json",
};
