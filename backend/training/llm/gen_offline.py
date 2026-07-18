"""Sinh dữ liệu MẪU offline (KHÔNG cần API/Claude) để debug pipeline fine-tune.

Dùng template + catalog tổng hợp (taxonomy.build_synthetic_catalog) để tạo cặp
comment→reply grounded, đúng schema raw record của gen_synthetic.py. Data này KHÔNG
thay thế chất lượng teacher thật, nhưng đủ để chạy thử end-to-end:
    gen_offline → build_dataset → train_qlora → merge_export → evaluate.

Mọi reply được dựng từ đúng dữ liệu catalog nên qua được bộ lọc grounding.

Cách chạy (từ backend/):
    python -m training.llm.gen_offline --n 1200 --out training/data/raw/offline_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

from modules.llm.rag import ProductStore
from .taxonomy import CODESWITCH_TERMS, INTENTS, build_synthetic_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("gen_offline")

_fmt = ProductStore._fmt_price  # định dạng giá y hệt lúc serve

_NAMES = ["Linh", "Trang", "Huy", "Vy", "Mai", "Nam", "Thảo", "Quân", "Ngọc",
          "Duy", "Hà", "Phúc", "An", "Hương", "Tú", "Bảo", "🌸Ngọc🌸", "cu Tí",
          "mẹ Bắp", "chị Hai", "Khách 123", "😍fan cứng😍"]


def _pick_name(rng: random.Random) -> str:
    return rng.choice(_NAMES)


def _price_str(p: dict[str, Any]) -> str:
    """Chuỗi giá để đưa vào reply (ưu tiên giá KM nếu có)."""
    return _fmt(p.get("gia_km") or p.get("gia"))


def _in_stock(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in products if (p.get("ton_kho") or 0) > 0]


def _codeswitch(text: str, rng: random.Random) -> str:
    """Chêm nhẹ 1 từ tiếng Anh vào cuối câu (giữ tiếng Việt làm nền)."""
    term = rng.choice(CODESWITCH_TERMS)
    inserts = {
        "size": " nha, mình xem size cho chuẩn",
        "freeship": ", bên shop có freeship luôn",
        "sale": " đang sale nha",
        "order": ", cả nhà order sớm kẻo hết",
        "check": ", shop check kho giúp mình",
        "inbox": ", mình inbox shop nha",
        "combo": ", có combo tiết kiệm nữa",
        "order ": ", order liền nha",
    }
    return text.rstrip(".! ") + inserts.get(term, f" {term} nha") + "!"


# ---- Reply builder cho từng intent ---------------------------------------
def _gen_gia_km(cat, rng):
    p = rng.choice(_in_stock(cat["products"]) or cat["products"])
    ten = p["ten"]
    c = rng.choice([f"{ten} bao nhiêu tiền vậy shop?", f"{ten} giá sao ạ?",
                    f"cho hỏi giá {ten} với", f"{ten} nhiêu z shop ơi?"])
    if p.get("gia_km"):
        r = (f"Dạ {ten} đang khuyến mãi còn {_fmt(p['gia_km'])} thôi cả nhà ơi, "
             f"mình chốt đơn shop gói gửi liền nha!")
    else:
        r = f"Dạ {ten} giá {_fmt(p['gia'])} nha cả nhà, mình chốt đơn shop gửi liền nhé!"
    return c, r


def _gen_ton_size_mau(cat, rng):
    p = rng.choice(_in_stock(cat["products"]) or cat["products"])
    ten = p["ten"]
    sizes = ", ".join(str(s) for s in p.get("size", []))
    mau = ", ".join(p.get("mau", []))
    c = rng.choice([f"{ten} còn hàng không shop?", f"{ten} còn size nào ạ?",
                    f"{ten} có màu gì shop ơi?"])
    r = (f"Dạ {ten} còn đủ hàng nha cả nhà, có size {sizes}, màu {mau}, "
         f"mình chốt màu với size shop gói liền ạ!")
    return c, r


def _gen_mo_ta_faq(cat, rng):
    p = rng.choice(cat["products"])
    ten = p["ten"]
    info = rng.choice(p.get("faq") or [p.get("mo_ta", "")])
    c = rng.choice([f"chất {ten} sao shop?", f"{ten} form thế nào ạ?",
                    f"{ten} mặc có đẹp không shop ơi?"])
    r = f"Dạ {info} nha cả nhà, mình cứ yên tâm đặt nhé!"
    return c, r


def _gen_chinh_sach(cat, rng):
    policies = (cat["shop"].get("policies") or {})
    if not policies:
        return _gen_gia_km(cat, rng)
    key = rng.choice(list(policies.keys()))
    val = policies[key]
    q = {"ship": "Ship về tỉnh mấy ngày shop?", "doi_tra": "Có đổi trả được không ạ?",
         "thanh_toan": "Thanh toán kiểu gì shop ơi?"}.get(key, "Chính sách shop sao ạ?")
    r = f"Dạ {val} nha cả nhà, mình cứ yên tâm đặt hàng nhé!"
    return q, r


def _gen_tu_van_size(cat, rng):
    p = rng.choice(_in_stock(cat["products"]) or cat["products"])
    ten = p["ten"]
    sizes = ", ".join(str(s) for s in p.get("size", []))
    kg = rng.choice([50, 55, 58, 60, 62, 65, 68])
    c = f"em nặng {kg}kg mặc {ten} size nào shop?"
    r = (f"Dạ {ten} bên mình có size {sizes} nha cả nhà, mình cho shop xin thêm "
         f"chiều cao để tư vấn size chuẩn nhất ạ!")
    return c, r


def _gen_so_sanh(cat, rng):
    if len(cat["products"]) < 2:
        return _gen_gia_km(cat, rng)
    a, b = rng.sample(cat["products"], 2)
    c = f"{a['ten']} với {b['ten']} cái nào đáng mua hơn shop?"
    r = (f"Dạ {a['ten']} giá {_price_str(a)}, còn {b['ten']} giá {_price_str(b)} nha cả nhà, "
         f"tùy nhu cầu mình chọn, shop tư vấn thêm nếu cần ạ!")
    return c, r


def _gen_chot_don(cat, rng):
    p = rng.choice(_in_stock(cat["products"]) or cat["products"])
    ten = p["ten"]
    c = rng.choice([f"chốt {ten} cho mình 1 cái", f"lấy {ten} nha shop",
                    f"đặt {ten} sao shop?"])
    r = (f"Dạ cả nhà cho shop xin tên, số điện thoại và địa chỉ để lên đơn {ten} "
         f"ngay nha, shop gói gửi liền ạ!")
    return c, r


def _gen_off_topic(cat, rng):
    c = rng.choice(["shop ăn cơm chưa 😄", "hôm nay trời đẹp nhỉ", "shop ở đâu vậy"])
    r = ("Dạ cả nhà đang quan tâm mẫu nào để shop tư vấn liền nha, "
         "hôm nay shop nhiều mẫu đẹp giá tốt lắm ạ!")
    return c, r


# ---- Hard-negatives (chống bịa) ------------------------------------------
def _gen_ngoai_catalog(cat, rng):
    fake = rng.choice(["iPhone 15", "nồi chiên không dầu", "sữa rửa mặt",
                       "tai nghe bluetooth", "đồng hồ Rolex"])
    have = ", ".join(p["ten"] for p in cat["products"][:3])
    c = f"shop có bán {fake} không ạ?"
    r = (f"Dạ shop chưa có {fake} nha cả nhà ơi, bên mình đang có {have}, "
         f"cả nhà tham khảo giúp shop nha!")
    return c, r


def _gen_gia_khong_co(cat, rng):
    p = rng.choice(cat["products"])
    ten = p["ten"]
    sizes = ", ".join(str(s) for s in p.get("size", []))
    weird = rng.choice(["6XL", "size 50", "màu dạ quang", "màu trong suốt"])
    c = f"{ten} có {weird} không shop?"
    r = (f"Dạ {ten} bên shop chỉ có size {sizes} thôi cả nhà ạ, "
         f"mình xem giúp shop lựa chọn đang có nha!")
    return c, r


def _gen_het_hang(cat, rng):
    # đảm bảo có 1 sản phẩm hết hàng, và catalog phản ánh đúng (để reply nhất quán)
    out = [p for p in cat["products"] if (p.get("ton_kho") or 0) == 0]
    if not out:
        products = list(cat["products"])
        idx = rng.randrange(len(products))
        p = {**products[idx], "ton_kho": 0}
        products[idx] = p
        cat = {"shop": cat["shop"], "products": products}
    else:
        p = rng.choice(out)
    ten = p["ten"]
    c = rng.choice([f"{ten} còn hàng không shop?", f"mua {ten} được không ạ?"])
    r = (f"Dạ {ten} hiện đang hết hàng cả nhà ơi, shop sẽ báo ngay khi có hàng lại, "
         f"mình để lại thông tin shop giữ chỗ nha!")
    return c, r, cat


def _gen_multi_turn(cat, rng):
    p = rng.choice(_in_stock(cat["products"]) or cat["products"])
    ten = p["ten"]
    name = _pick_name(rng)
    conv = [
        {"role": "user", "author": name, "text": f"{ten} còn hàng không shop?"},
        {"role": "assistant", "text": f"Dạ {ten} còn hàng nha cả nhà, giá {_price_str(p)}, mình chốt shop gói liền ạ!"},
        {"role": "user", "author": name, "text": "ok chốt cho mình 1 cái"},
        {"role": "assistant", "text": "Dạ cả nhà cho shop xin tên, số điện thoại và địa chỉ để lên đơn ngay nha!"},
    ]
    return conv


_BUILDERS: dict[str, Callable] = {
    "gia_km": _gen_gia_km,
    "ton_kho_size_mau": _gen_ton_size_mau,
    "mo_ta_faq": _gen_mo_ta_faq,
    "chinh_sach": _gen_chinh_sach,
    "tu_van_size": _gen_tu_van_size,
    "so_sanh": _gen_so_sanh,
    "chot_don": _gen_chot_don,
    "off_topic": _gen_off_topic,
    "ngoai_catalog": _gen_ngoai_catalog,
    "gia_khong_co": _gen_gia_khong_co,
    # het_hang & multi_turn xử lý riêng
}


def _weighted_intents(rng: random.Random, n: int) -> list[dict[str, Any]]:
    weights = [i["weight"] for i in INTENTS]
    return rng.choices(INTENTS, weights=weights, k=n)


def _make_record(intent, cat, rng, codeswitch):
    name = _pick_name(rng)
    if intent["name"] == "multi_turn_chot":
        conv = _gen_multi_turn(cat, rng)
        return {"shop": cat["shop"], "products": cat["products"], "intent": intent["name"],
                "hard_negative": False, "codeswitch": False, "conversation": conv}
    if intent["name"] == "het_hang":
        c, r, cat = _gen_het_hang(cat, rng)
    else:
        builder = _BUILDERS.get(intent["name"], _gen_gia_km)
        c, r = builder(cat, rng)
    if codeswitch and not intent["hard_negative"]:
        r = _codeswitch(r, rng)
    conv = [{"role": "user", "author": name, "text": c}, {"role": "assistant", "text": r}]
    return {"shop": cat["shop"], "products": cat["products"], "intent": intent["name"],
            "hard_negative": intent["hard_negative"], "codeswitch": codeswitch,
            "conversation": conv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="số mẫu muốn sinh")
    ap.add_argument("--out", default="training/data/raw/offline_raw.jsonl")
    ap.add_argument("--codeswitch-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--products", default=None,
                    help="products.json thật để trộn vào catalog (tùy chọn)")
    ap.add_argument("--real-ratio", type=float, default=0.25)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    real = None
    if args.products and Path(args.products).exists():
        d = json.loads(Path(args.products).read_text(encoding="utf-8"))
        real = {"shop": d.get("shop", {}), "products": d.get("products", [])}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for intent in _weighted_intents(rng, args.n):
            cat = real if (real and rng.random() < args.real_ratio) else build_synthetic_catalog(rng)
            cat = {"shop": cat["shop"], "products": list(cat["products"])}
            codeswitch = rng.random() < args.codeswitch_ratio
            rec = _make_record(intent, cat, rng, codeswitch)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    logger.info("Sinh offline %d mẫu → %s", written, out)


if __name__ == "__main__":
    main()
