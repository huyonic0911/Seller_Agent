# Live2D models

Mặc định app dùng avatar SVG (không cần model). Để bật Live2D thật:

1. Tải một model Cubism 4 miễn phí, ví dụ **Hiyori** từ bộ sample chính thức của Live2D
   (https://www.live2d.com/en/learn/sample/) hoặc các model free khác.
2. Giải nén vào thư mục này, ví dụ: `public/models/hiyori/hiyori_pro_t11.model3.json` (kèm textures, `.moc3`, `.motion3.json`).
3. Tạo file `frontend/.env`:

   ```
   VITE_USE_LIVE2D=true
   VITE_LIVE2D_MODEL=/models/hiyori/hiyori_pro_t11.model3.json
   ```

4. Chạy lại `npm run dev`.

> Cubism Core runtime đã được nhúng qua `<script>` trong `index.html`.
> Lip-sync điều khiển tham số `ParamMouthOpenY` — tên tham số có thể khác tùy model,
> chỉnh trong `src/lib/live2d.ts` nếu cần.
