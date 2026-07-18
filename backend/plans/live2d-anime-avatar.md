# Kế hoạch: Nhân vật anime Live2D biểu cảm cho Seller Agent

## Context

Dự án AI Virtual Assistant gồm 3 phần: LLM sinh câu trả lời, viXTTS đọc tiếng, và
nhân vật anime tương tác với người dùng. Phần **LLM + TTS + pipeline WebSocket đã
hoạt động**; phần avatar cũng đã có nền móng trong `frontend/`:

- Live2D (Cubism 4) đã scaffold qua `pixi-live2d-display` — `frontend/src/lib/live2d.ts`
  + `frontend/src/components/Live2DStage.tsx`, bật bằng `VITE_USE_LIVE2D=true`.
- Avatar SVG fallback (anime girl) luôn chạy được.
- Lip-sync đã chạy: phân tích RMS audio → `ParamMouthOpenY` / `scaleY` miệng
  (`frontend/src/lib/lipsync.ts`).

Cần bổ sung: (1) một model Live2D thật + làm nhân vật "sống" (chớp mắt, thở, lắc đầu
khi idle), (2) **biểu cảm theo cảm xúc** — backend gắn nhãn cảm xúc vào mỗi câu trả
lời, avatar đổi biểu cảm/cử chỉ tương ứng.

Quyết định đã chốt với người dùng:
- Hướng **Live2D (2.5D)** — phát triển tiếp scaffold sẵn có.
- Mức độ: **lip-sync + idle sống động + biểu cảm theo cảm xúc**.
- Model: **dùng model free (Hiyori) để chạy thông pipeline trước**, đổi model riêng sau.

## Nguyên tắc quan trọng

`backend/app/prompt.py` là **nguồn sự thật chung giữa train và serve** (fine-tune viXTTS/LLM
phụ thuộc). **KHÔNG chèn tag cảm xúc vào prompt** → sẽ lệch contract train/serve. Thay vào
đó suy luận cảm xúc từ *text câu trả lời đã sinh* bằng một module Python riêng — cách này
hoạt động với mọi provider (qwen / anthropic / finetuned / offline) và không đụng contract.

## Backend — thêm nhãn cảm xúc

### 1. Module cảm xúc mới: `backend/app/emotion.py`
- Định nghĩa tập cảm xúc cố định: `neutral | happy | excited | thinking | apologetic | friendly`.
- Hàm thuần `infer_emotion(text: str) -> str`: suy luận bằng heuristic từ khóa tiếng Việt
  (vd "hết hàng", "xin lỗi" → `apologetic`; "khuyến mãi", "giảm giá", "🎉", "!" nhiều → `excited`;
  "dạ", "nha", "ạ" lịch sự → `friendly`; câu hỏi/không match → `neutral`/`thinking`).
- Pure function, không phụ thuộc settings → dễ test và tái dùng.

### 2. Gắn cảm xúc vào message reply: `backend/app/pipeline.py`
- Trong `_handle()` (dòng ~105 sau khi có `reply`): gọi `emotion = infer_emotion(reply)`.
- Thêm `"emotion": emotion` vào cả broadcast `{"type": "reply", ...}` (dòng 119-128)
  và broadcast `{"type": "error", ...}` (dùng `"apologetic"`).

Không sửa `llm.py` và `prompt.py`.

## Frontend — Live2D biểu cảm + idle sống động

### 3. Kiểu message: `frontend/src/hooks/useWebSocket.ts`
- Thêm `emotion?: string` vào nhánh `reply` (và `error`) của `StreamMessage`.

### 4. Mở rộng controller Live2D: `frontend/src/lib/live2d.ts`
- Mở rộng interface `Live2DController` thêm `setEmotion(name: string)` (bên cạnh `setMouth`).
- **Idle sống động**: kích hoạt idle motion group của model + đảm bảo auto-blink & breath
  (Hiyori có sẵn motion/physics; nếu thiếu, drive thủ công `ParamEyeLOpen/ParamEyeROpen`
  theo chu kỳ ngẫu nhiên và `ParamBreath` bằng sine trong ticker hiện có).
- **Biểu cảm**: map tên cảm xúc → expression file `.exp3.json` của model (qua
  `internalModel.motionManager.expressionManager` / API `model.expression(name)` của
  pixi-live2d-display). Có bảng map `EMOTION_TO_EXPRESSION` với fallback về `neutral` nếu
  model thiếu expression đó (Hiyori sample có sẵn vài expression; nếu không đủ, dùng
  head-angle/tham số làm biểu cảm nhẹ).

### 5. Truyền cảm xúc xuống stage: `frontend/src/components/Live2DStage.tsx`
- Thêm prop `emotion?: string`. Khi đổi: gọi `ctrl.setEmotion(emotion)` (Live2D) hoặc
  đổi hình mắt/miệng của `SvgAvatar` (fallback) — vd nhíu mày khi `apologetic`, mắt cong
  khi `happy`. Trở về `neutral` khi ngừng nói.

### 6. Nối cảm xúc ở App: `frontend/src/App.tsx`
- `queueRef` lưu thêm `emotion`; trong `drain()` set `emotion` state trước khi `lipSync.play`,
  reset về `neutral` sau khi nói xong; truyền `emotion` vào `<Live2DStage ... />`.

### 7. Asset model + bật Live2D
- Tải model Hiyori free (Cubism 4 sample chính thức) vào
  `frontend/public/models/hiyori/` (kèm `.moc3`, textures, `.motion3.json`, `.exp3.json`).
- Tạo `frontend/.env`: `VITE_USE_LIVE2D=true` và `VITE_LIVE2D_MODEL=/models/hiyori/hiyori_pro_t11.model3.json`.
- Kiểm tra `index.html` đã nhúng `live2dcubismcore.min.js` (theo comment là đã có; xác nhận,
  nếu chưa thì thêm — đây là runtime Cubism Core, tải từ trang Live2D).

## Files sẽ sửa/tạo
- Tạo: `backend/app/emotion.py`, `frontend/.env`, `frontend/public/models/hiyori/*`
- Sửa: `backend/app/pipeline.py`, `frontend/src/hooks/useWebSocket.ts`,
  `frontend/src/lib/live2d.ts`, `frontend/src/components/Live2DStage.tsx`,
  `frontend/src/App.tsx` (và có thể `index.html` nếu thiếu Cubism Core).

## Verification (chạy end-to-end)
1. Backend: `cd backend && uvicorn app.main:app --reload` — kiểm `GET /health` OK.
2. Test nhanh module cảm xúc: `python -c "from app.emotion import infer_emotion; print(infer_emotion('Dạ hết hàng rồi ạ'))"` → `apologetic`.
3. Frontend: `cd frontend && npm run dev`, mở trang, bấm "Bật âm thanh".
4. Gửi comment từ AdminPanel (vd hỏi sản phẩm khuyến mãi vs hỏi sản phẩm hết hàng):
   - Xác nhận avatar Live2D Hiyori hiển thị (không rơi về SVG) và **nhép miệng theo tiếng đọc**.
   - Xác nhận **biểu cảm đổi theo cảm xúc** (vui khi khuyến mãi, buồn/xin lỗi khi hết hàng).
   - Xác nhận khi idle avatar vẫn **chớp mắt / thở / lắc đầu nhẹ**.
5. Test fallback: đặt `VITE_USE_LIVE2D=false` — SVG avatar vẫn nhép miệng + đổi biểu cảm cơ bản.

## Mở rộng sau (ngoài phạm vi hiện tại)
- Lip-sync theo viseme/phoneme thay vì biên độ RMS (chính xác khẩu hình hơn).
- Cảm xúc do LLM chủ động quyết định (structured output) thay vì heuristic — chỉ khi tách
  được khỏi prompt contract của train.
- Model nhân vật riêng (VRoid/Cubism/commission) thay Hiyori.
