# Avatar models (VRM)

App render avatar bằng **VRM** (three.js + `@pixiv/three-vrm`) — nhân vật anime 3D
toon (2.5D). File `.vrm` **không commit** vào git (nặng, xem `.gitignore`).

## Dùng nhân vật của bạn (khuyến nghị)

1. Tải **VRoid Studio** (miễn phí): https://vroid.com/en/studio
2. Thiết kế cô gái anime theo ý bạn, rồi **Export → VRM**.
3. Lưu file vào đây với tên `avatar.vrm`:

   ```
   frontend/public/models/avatar.vrm
   ```

4. (Tùy chọn) trỏ đường dẫn khác qua `frontend/.env`:

   ```
   VITE_VRM_MODEL=/models/ten-file.vrm
   ```

5. Chạy lại `npm run dev`.

## Ghi chú kỹ thuật

- Lip-sync điều khiển expression preset **`aa`** (độ mở miệng) theo biên độ audio.
- Cảm xúc map 6 nhãn backend → preset VRM (happy/sad/relaxed/surprised…), xem
  `EMOTION_TO_VRM` trong `src/lib/vrm.ts`.
- Chớp mắt / thở / lắc đầu nhẹ được drive thủ công trong `src/lib/vrm.ts`.
- VRM nên có đủ các expression preset chuẩn để cảm xúc & khẩu hình hoạt động —
  VRoid Studio export sẵn các preset này.
