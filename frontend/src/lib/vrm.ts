// Bộ render avatar VRM (three.js + @pixiv/three-vrm). Thay cho Live2D cũ.
//
// Nhân vật thiết kế trong VRoid Studio → export .vrm (shader MToon = nhìn 2.5D anime).
// Controller cung cấp:
//   - setMouth(level)   : độ mở miệng 0..1 (lip-sync theo biên độ audio)
//   - setEmotion(name)  : 6 nhãn cảm xúc từ backend → preset expression VRM
//   - destroy()
//
// Idle "sống động" được drive thủ công (three-vrm không tự chớp mắt): chớp mắt ngẫu
// nhiên, thở (sine vào bone spine), lắc/nghiêng đầu nhẹ, mắt nhìn quanh.

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  VRM,
  VRMLoaderPlugin,
  VRMUtils,
  VRMExpressionPresetName,
  VRMHumanBoneName,
} from "@pixiv/three-vrm";

// Các action điều khiển bằng nút (xoay xương/expression, không cần file animation).
export type ActionName =
  | "closeEyes" // nhắm mắt (bật/tắt, giữ nguyên đến khi bấm lại)
  | "raiseHand" // giơ tay (giữ ~2s)
  | "wave" // vẫy tay
  | "clap" // vỗ tay
  | "nod" // gật đầu
  | "shake"; // lắc đầu

export interface AvatarController {
  setMouth(level: number): void;
  setEmotion(name: string): void;
  playAction(name: ActionName): void;
  zoom(delta: number): void; // delta<0: lại gần (zoom in); delta>0: ra xa (zoom out)
  destroy(): void;
}

// Tư thế tay (radian) trên rig CHUẨN HOÁ của VRM — gốc là T-pose (tay giang ngang),
// nên phải chủ động hạ tay xuống MỖI FRAME, nếu không nhân vật luôn giang 2 tay.
// Quy ước trục z (đã canh theo model hiện tại): tay PHẢI z DƯƠNG = hạ xuống /
// z ÂM = giơ lên; tay TRÁI ngược dấu. Nếu đổi model mà tay sai hướng, đổi dấu 2 số này.
const ARM_DOWN = 1.15; // góc hạ tay ở tư thế nghỉ (A-pose) — tay phải = +ARM_DOWN
const RIGHT_UP = -1.4; // tay phải giơ thẳng lên

// 6 nhãn cảm xúc backend (core/emotion.py) → trọng số các preset expression VRM.
// Preset khả dụng: happy | angry | sad | relaxed | surprised | neutral.
const EMOTION_TO_VRM: Record<string, Partial<Record<string, number>>> = {
  neutral: {},
  happy: { happy: 0.8 },
  excited: { happy: 1.0, surprised: 0.3 },
  thinking: { relaxed: 0.5 },
  apologetic: { sad: 0.8 },
  friendly: { happy: 0.5 },
};

const EMOTION_PRESETS = ["happy", "angry", "sad", "relaxed", "surprised"] as const;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export async function initVRM(
  canvas: HTMLCanvasElement,
  modelUrl: string,
): Promise<AvatarController> {
  const parent = canvas.parentElement ?? undefined;
  const width = parent?.clientWidth || canvas.clientWidth || 480;
  const height = parent?.clientHeight || canvas.clientHeight || 640;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(28, width / height, 0.1, 20);
  camera.position.set(0, 1.3, 1.6); // khung cận nửa thân trên

  // Ánh sáng cho MToon đổ shade (toon).
  const dir = new THREE.DirectionalLight(0xffffff, 1.4);
  dir.position.set(1, 1.6, 1.2);
  scene.add(dir);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.9));

  // Nạp model VRM.
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(modelUrl);
  const vrm = gltf.userData.vrm as VRM;

  // Tối ưu + quay mặt về camera (VRM0 nhìn -Z; hàm này chỉ xoay khi là VRM0).
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  scene.add(vrm.scene);

  // Căn camera nhìn vào đầu nhân vật.
  const head = vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head);
  const headPos = new THREE.Vector3();
  if (head) {
    head.getWorldPosition(headPos);
  } else {
    headPos.set(0, 1.35, 0);
  }
  // Khung hình: tâm hạ xuống ngực/eo, camera lùi xa để thấy tới eo.
  // Muốn zoom XA hơn → tăng CAMERA_DIST; thấy THẤP hơn (tới hông) → tăng FRAME_DROP.
  const CAMERA_DIST = 2.3; // khoảng cách camera mặc định (m)
  const FRAME_DROP = 0.45; // hạ tâm khung xuống dưới đầu (m)
  const ZOOM_MIN = 1.1; // gần nhất (cận mặt)
  const ZOOM_MAX = 4.5; // xa nhất (cả người)
  const centerY = headPos.y - FRAME_DROP;
  let camDist = CAMERA_DIST;
  const applyCamera = () => {
    camera.position.set(headPos.x, centerY, headPos.z + camDist);
    camera.lookAt(headPos.x, centerY, headPos.z);
  };
  applyCamera();

  // Mắt nhìn theo 1 target đặt trước mặt (VRM lookAt).
  const lookTarget = new THREE.Object3D();
  lookTarget.position.set(headPos.x, headPos.y, headPos.z + 1.5);
  scene.add(lookTarget);
  if (vrm.lookAt) vrm.lookAt.target = lookTarget;

  const em = vrm.expressionManager;
  const hb = (n: VRMHumanBoneName) => vrm.humanoid?.getNormalizedBoneNode(n) ?? null;
  const spine = hb(VRMHumanBoneName.Spine);
  const neck = hb(VRMHumanBoneName.Neck);
  const spineRestX = spine?.rotation.x ?? 0;
  const neckRestX = neck?.rotation.x ?? 0;

  // Xương tay cho tư thế nghỉ (A-pose) + gesture.
  const rUpper = hb(VRMHumanBoneName.RightUpperArm);
  const rLower = hb(VRMHumanBoneName.RightLowerArm);
  const lUpper = hb(VRMHumanBoneName.LeftUpperArm);
  const lLower = hb(VRMHumanBoneName.LeftLowerArm);

  // Xương ngón tay (Index/Middle/Ring/Little × Proximal/Intermediate/Distal) để
  // nắm hờ cho bàn tay tự nhiên. `sign` xử lý đối xứng trái/phải.
  const fingerBones: { node: THREE.Object3D; sign: number }[] = [];
  for (const [side, sign] of [
    ["Right", 1],
    ["Left", -1],
  ] as const) {
    for (const f of ["Index", "Middle", "Ring", "Little"] as const) {
      for (const s of ["Proximal", "Intermediate", "Distal"] as const) {
        const key = `${side}${f}${s}` as keyof typeof VRMHumanBoneName;
        const node = hb(VRMHumanBoneName[key]);
        if (node) fingerBones.push({ node, sign });
      }
    }
  }
  // Độ cong ngón khi nghỉ (radian). Nếu ngón cong NGƯỢC ra sau, ĐỔI DẤU hằng số này.
  const FINGER_CURL = 0.45;

  // --- trạng thái động ---
  let mouthTarget = 0;
  let mouth = 0;
  let emotionTarget: Partial<Record<string, number>> = {};
  const emotionCur: Record<string, number> = {};
  for (const p of EMOTION_PRESETS) emotionCur[p] = 0;

  // chớp mắt
  let blink = 0;
  let nextBlink = 0.8 + Math.random() * 3;
  let blinkT = -1; // -1 = không chớp; >=0 = đang trong chuỗi chớp
  const BLINK_DUR = 0.12;

  // action (gesture) theo nút
  let eyesClosed = false; // toggle nhắm mắt
  let action: ActionName | null = null;
  let actionT = 0;
  const ACTION_DUR: Record<ActionName, number> = {
    closeEyes: 0, // xử lý riêng (toggle)
    raiseHand: 2.0,
    wave: 2.2,
    clap: 2.2,
    nod: 1.2,
    shake: 1.2,
  };
  // Bao hình đóng/mở (0→1→0): ramp lên 25%, giữ, ramp xuống 25%.
  const envelope = (p: number) => (p < 0.25 ? p / 0.25 : p > 0.75 ? (1 - p) / 0.25 : 1);

  const clock = new THREE.Clock();
  let elapsed = 0;

  const onResize = () => {
    const w = parent?.clientWidth || width;
    const h = parent?.clientHeight || height;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener("resize", onResize);

  const animate = () => {
    const dt = Math.min(clock.getDelta(), 0.05);
    elapsed += dt;

    // Miệng: bám nhanh về target (lip-sync).
    mouth = lerp(mouth, mouthTarget, 1 - Math.exp(-dt * 25));

    // Cảm xúc: chuyển mượt về target.
    for (const p of EMOTION_PRESETS) {
      emotionCur[p] = lerp(emotionCur[p], emotionTarget[p] ?? 0, 1 - Math.exp(-dt * 6));
    }

    // Chớp mắt ngẫu nhiên.
    if (blinkT < 0) {
      nextBlink -= dt;
      if (nextBlink <= 0) blinkT = 0;
    } else {
      blinkT += dt;
      const half = BLINK_DUR / 2;
      blink = blinkT < half ? blinkT / half : Math.max(0, 1 - (blinkT - half) / half);
      if (blinkT >= BLINK_DUR) {
        blink = 0;
        blinkT = -1;
        nextBlink = 1.5 + Math.random() * 4;
      }
    }

    if (em) {
      em.setValue(VRMExpressionPresetName.Aa, mouth);
      // Nhắm mắt (toggle) đè lên chớp mắt tự động.
      em.setValue(VRMExpressionPresetName.Blink, eyesClosed ? 1 : blink);
      for (const p of EMOTION_PRESETS) {
        em.setValue(p as VRMExpressionPresetName, emotionCur[p]);
      }
    }

    // Idle: thở (spine), lắc đầu nhẹ (neck), mắt nhìn quanh.
    if (spine) spine.rotation.x = spineRestX + Math.sin(elapsed * 1.6) * 0.015;
    if (neck) {
      neck.rotation.y = Math.sin(elapsed * 0.5) * 0.06;
      neck.rotation.z = Math.sin(elapsed * 0.37) * 0.03;
    }
    lookTarget.position.x = headPos.x + Math.sin(elapsed * 0.3) * 0.15;
    lookTarget.position.y = headPos.y + Math.sin(elapsed * 0.23) * 0.08;

    // Tư thế tay NGHỈ (A-pose) mỗi frame — rig chuẩn hoá mặc định là T-pose giang ngang.
    // Gesture bên dưới sẽ ghi đè khi đang chạy; hết action tay tự về A-pose.
    if (rUpper) rUpper.rotation.set(0, 0, ARM_DOWN);
    if (lUpper) lUpper.rotation.set(0, 0, -ARM_DOWN);
    if (rLower) rLower.rotation.set(0, 0, 0);
    if (lLower) lLower.rotation.set(0, 0, 0);

    // Nắm hờ các ngón tay để bàn tay tự nhiên (không xoè thẳng đơ).
    for (const { node, sign } of fingerBones) node.rotation.z = sign * FINGER_CURL;

    // Gesture theo nút (nội suy từ A-pose bằng bao hình w = 0→1→0).
    if (action && action !== "closeEyes") {
      actionT += dt;
      const dur = ACTION_DUR[action];
      const w = envelope(Math.min(1, actionT / dur));
      switch (action) {
        case "raiseHand":
          if (rUpper) rUpper.rotation.z = lerp(ARM_DOWN, RIGHT_UP, w);
          break;
        case "wave":
          if (rUpper) rUpper.rotation.z = lerp(ARM_DOWN, RIGHT_UP - 0.1, w);
          if (rLower) rLower.rotation.z = w * 0.45 * Math.sin(actionT * 13); // vẫy cẳng tay
          break;
        case "clap": {
          // đưa hai tay ra trước ngực (nghiêng vai + khép vào giữa), gập khuỷu, vỗ qua lại
          const osc = w * 0.22 * Math.abs(Math.sin(actionT * 11));
          if (rUpper) rUpper.rotation.set(-0.5 * w, -0.5 * w, lerp(ARM_DOWN, 0.35, w) - osc);
          if (lUpper) lUpper.rotation.set(-0.5 * w, 0.5 * w, lerp(-ARM_DOWN, -0.35, w) + osc);
          if (rLower) rLower.rotation.set(0, 1.2 * w, 0); // gập khuỷu, cẳng tay hướng vào giữa
          if (lLower) lLower.rotation.set(0, -1.2 * w, 0);
          break;
        }
        case "nod":
          if (neck) neck.rotation.x = neckRestX + w * 0.3 * Math.sin(actionT * 9);
          break;
        case "shake":
          if (neck) neck.rotation.y += w * 0.4 * Math.sin(actionT * 9);
          break;
      }
      if (actionT >= dur) {
        action = null;
        actionT = 0;
      }
    }

    vrm.update(dt);
    renderer.render(scene, camera);
  };
  renderer.setAnimationLoop(animate);

  return {
    setMouth(level: number) {
      mouthTarget = Math.min(1, Math.max(0, level));
    },
    setEmotion(name: string) {
      emotionTarget = EMOTION_TO_VRM[name] ?? {};
    },
    playAction(name: ActionName) {
      if (name === "closeEyes") {
        eyesClosed = !eyesClosed; // bật/tắt
        return;
      }
      action = name; // baseline A-pose mỗi frame tự lo việc reset tay
      actionT = 0;
    },
    zoom(delta: number) {
      camDist = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, camDist + delta));
      applyCamera();
    },
    destroy() {
      window.removeEventListener("resize", onResize);
      renderer.setAnimationLoop(null);
      VRMUtils.deepDispose(vrm.scene);
      renderer.dispose();
    },
  };
}
