// Bộ nạp Live2D (tùy chọn). Dùng dynamic import để nếu chưa cài model / Cubism Core
// thì phần còn lại của app vẫn chạy (fallback avatar SVG).
//
// Yêu cầu để bật:
//   1. VITE_USE_LIVE2D=true
//   2. Có <script live2dcubismcore.min.js> trong index.html (đã có sẵn)
//   3. Đặt model Cubism vào public/models/... và trỏ VITE_LIVE2D_MODEL

export interface Live2DController {
  setMouth(level: number): void;
  destroy(): void;
}

export async function initLive2D(
  canvas: HTMLCanvasElement,
  modelUrl: string,
): Promise<Live2DController> {
  const PIXI = await import("pixi.js");
  const { Live2DModel } = await import("pixi-live2d-display");

  // pixi-live2d-display cần PIXI global + ticker.
  (window as any).PIXI = PIXI;
  Live2DModel.registerTicker(PIXI.Ticker as any);

  const app = new PIXI.Application({
    view: canvas,
    resizeTo: canvas.parentElement ?? undefined,
    backgroundAlpha: 0,
    antialias: true,
    autoStart: true,
  });

  const model = await Live2DModel.from(modelUrl, { autoInteract: false });
  app.stage.addChild(model as any);

  const fit = () => {
    const w = app.renderer.width;
    const h = app.renderer.height;
    const scale = Math.min(w / model.width, h / model.height) * 0.9;
    model.scale.set(scale);
    model.anchor?.set?.(0.5, 0.5);
    model.position.set(w / 2, h / 2);
  };
  fit();
  window.addEventListener("resize", fit);

  let mouth = 0;
  const core = (model.internalModel as any)?.coreModel;
  // Ghi đè giá trị miệng mỗi frame (chạy sau update của model → giá trị của ta thắng khi render).
  const ticker = () => {
    try {
      core?.setParameterValueById?.("ParamMouthOpenY", mouth);
    } catch {
      /* model không có param này */
    }
  };
  app.ticker.add(ticker);

  return {
    setMouth(level: number) {
      mouth = Math.min(1, Math.max(0, level));
    },
    destroy() {
      window.removeEventListener("resize", fit);
      app.ticker.remove(ticker);
      app.destroy(true, { children: true });
    },
  };
}
