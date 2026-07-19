// Cấu hình frontend (đọc từ biến môi trường Vite, có mặc định hợp lý).

const BACKEND_HOST = import.meta.env.VITE_BACKEND_HOST ?? "localhost:8000";

export const config = {
  // WebSocket tới backend
  wsStream: `ws://${BACKEND_HOST}/ws/stream`,
  wsComments: `ws://${BACKEND_HOST}/ws/comments`,
  httpBase: `http://${BACKEND_HOST}`,

  // Avatar VRM (three.js + three-vrm). Thiết kế nhân vật trong VRoid Studio, export
  // .vrm rồi đặt vào frontend/public/models/ và trỏ VITE_VRM_MODEL tới file đó.
  vrmModel: import.meta.env.VITE_VRM_MODEL ?? "/models/avatar.vrm",
};
