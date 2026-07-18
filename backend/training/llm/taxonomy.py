"""Taxonomy intent + bộ sinh catalog tổng hợp cho dữ liệu fine-tune.

Dùng chung bởi gen_synthetic (đặt quota theo intent), build_dataset (chia eval
stratified theo intent) và evaluate (báo cáo theo intent).
"""
from __future__ import annotations

import random
from typing import Any

# --- Taxonomy intent -------------------------------------------------------
# hard_negative=True: câu hỏi mà thông tin KHÔNG có trong catalog → reply đúng
# phải là từ chối/nói thật + gợi ý cái đang có (tín hiệu chống bịa).
INTENTS: list[dict[str, Any]] = [
    {"name": "gia_km", "weight": 14, "hard_negative": False,
     "desc": "Hỏi giá, giá khuyến mãi, có đang sale không."},
    {"name": "ton_kho_size_mau", "weight": 14, "hard_negative": False,
     "desc": "Hỏi còn hàng không, còn size/màu nào."},
    {"name": "mo_ta_faq", "weight": 12, "hard_negative": False,
     "desc": "Hỏi chất liệu, form, thông tin trong mô tả/FAQ."},
    {"name": "chinh_sach", "weight": 8, "hard_negative": False,
     "desc": "Hỏi ship, freeship, đổi trả, thanh toán (chính sách shop)."},
    {"name": "tu_van_size", "weight": 8, "hard_negative": False,
     "desc": "Xin tư vấn chọn size theo cân nặng/số đo."},
    {"name": "so_sanh", "weight": 6, "hard_negative": False,
     "desc": "So sánh 2 sản phẩm trong catalog."},
    {"name": "chot_don", "weight": 8, "hard_negative": False,
     "desc": "Khách muốn đặt/chốt đơn; reply chốt đơn + xin thông tin."},
    {"name": "off_topic", "weight": 6, "hard_negative": False,
     "desc": "Câu không liên quan sản phẩm; lịch sự rồi kéo về bán hàng."},
    # --- Hard negatives (chống bịa) ---
    {"name": "ngoai_catalog", "weight": 9, "hard_negative": True,
     "desc": "Hỏi sản phẩm KHÔNG có trong catalog (vd iPhone, đồ shop không bán)."},
    {"name": "gia_khong_co", "weight": 5, "hard_negative": True,
     "desc": "Hỏi thuộc tính không tồn tại (size/màu SP không có, SP không list)."},
    {"name": "het_hang", "weight": 5, "hard_negative": True,
     "desc": "Hỏi mua SP đang ton_kho=0 → phải nói thật đã hết + hẹn về hàng."},
    {"name": "multi_turn_chot", "weight": 5, "hard_negative": False,
     "desc": "Hội thoại 2-4 lượt: hỏi → hỏi tiếp → chốt đơn."},
]

INTENT_NAMES = [i["name"] for i in INTENTS]

# Thuật ngữ tiếng Anh hay bị chêm trong livestream bán hàng VN.
CODESWITCH_TERMS = [
    "size", "freeship", "sale", "order", "oversize", "unisex", "canvas",
    "sneaker", "size up", "check", "inbox", "combo", "outfit", "basic",
]

# --- Bộ sinh catalog tổng hợp (đa dạng vượt 4 SKU gốc) --------------------
_PRODUCT_TEMPLATES = [
    {"prefix": "SP", "ten": "Áo thun cotton unisex", "gia": (99000, 199000),
     "mau": ["trắng", "đen", "be", "xanh navy", "xám"], "size": ["S", "M", "L", "XL"],
     "mo_ta": "Áo thun cotton 100%, form rộng, thấm hút tốt.",
     "faq": ["Size M mặc vừa người 55-65kg.", "Không xù lông sau khi giặt."]},
    {"prefix": "SP", "ten": "Quần jean nữ ống suông", "gia": (259000, 390000),
     "mau": ["xanh nhạt", "xanh đậm", "đen"], "size": ["27", "28", "29", "30"],
     "mo_ta": "Quần jean lưng cao ống suông, tôn dáng, co giãn nhẹ.",
     "faq": ["Size 28 phù hợp eo 70-72cm.", "Vải dày vừa, mặc 4 mùa."]},
    {"prefix": "SP", "ten": "Túi tote canvas", "gia": (69000, 129000),
     "mau": ["kem", "đen", "xanh rêu"], "size": ["Freesize"],
     "mo_ta": "Túi tote canvas dày, đựng laptop 14 inch, in tối giản.",
     "faq": ["Có khóa kéo bên trong.", "Số lượng có hạn."]},
    {"prefix": "SP", "ten": "Giày sneaker trắng", "gia": (350000, 490000),
     "mau": ["trắng", "kem"], "size": ["36", "37", "38", "39", "40", "41", "42"],
     "mo_ta": "Sneaker basic đế cao su êm chân, dễ phối đồ.",
     "faq": ["Nên tăng 1 size nếu chân to bề ngang."]},
    {"prefix": "SP", "ten": "Váy hoa nhí dáng suông", "gia": (199000, 350000),
     "mau": ["hồng", "vàng", "xanh mint"], "size": ["S", "M", "L"],
     "mo_ta": "Váy hoa nhí vải đũi mát, dáng suông che khuyết điểm.",
     "faq": ["Có lớp lót bên trong.", "Vải không nhăn nhiều."]},
    {"prefix": "SP", "ten": "Áo khoác gió unisex", "gia": (199000, 320000),
     "mau": ["đen", "xanh navy", "đỏ đô"], "size": ["M", "L", "XL"],
     "mo_ta": "Áo khoác gió 2 lớp cản gió nhẹ, có mũ.",
     "faq": ["Chống nước nhẹ.", "Gấp gọn bỏ túi được."]},
    {"prefix": "SP", "ten": "Mũ bucket vải kaki", "gia": (49000, 99000),
     "mau": ["be", "đen", "xanh"], "size": ["Freesize"],
     "mo_ta": "Mũ bucket kaki dày, vành vừa, chống nắng.",
     "faq": ["Giặt máy được."]},
    {"prefix": "SP", "ten": "Chân váy tennis xếp ly", "gia": (149000, 250000),
     "mau": ["đen", "trắng", "be"], "size": ["S", "M", "L"],
     "mo_ta": "Chân váy xếp ly có quần lót trong, năng động.",
     "faq": ["Cạp chun co giãn."]},
]

_SHOP_NAMES = ["Shop Xinh Store", "Nhà May Mắn Store", "Boutique Cỏ", "Local Brand Nắng"]


def build_synthetic_catalog(rng: random.Random) -> dict[str, Any]:
    """Sinh 1 catalog ngẫu nhiên đúng schema products.json (kèm 1 SP hết hàng)."""
    n = rng.randint(3, 6)
    templates = rng.sample(_PRODUCT_TEMPLATES, k=min(n, len(_PRODUCT_TEMPLATES)))
    products: list[dict[str, Any]] = []
    for i, t in enumerate(templates, start=1):
        gia = rng.randrange(t["gia"][0], t["gia"][1], 10000)
        has_km = rng.random() < 0.6
        gia_km = gia - rng.randrange(10000, 60000, 10000) if has_km else None
        if gia_km is not None and gia_km <= 0:
            gia_km = None
        # ~1/5 sản phẩm hết hàng để có case ton_kho=0 thật.
        ton = 0 if rng.random() < 0.2 else rng.randint(3, 150)
        products.append({
            "id": f"{t['prefix']}{i:03d}",
            "ten": t["ten"],
            "gia": gia,
            "gia_km": gia_km,
            "mau": rng.sample(t["mau"], k=rng.randint(1, len(t["mau"]))),
            "size": list(t["size"]),
            "ton_kho": ton,
            "mo_ta": t["mo_ta"],
            "faq": list(t["faq"]),
        })
    shop = {
        "name": rng.choice(_SHOP_NAMES),
        "policies": {
            "ship": "Freeship đơn từ 300k. Nội thành 2 ngày, tỉnh 3-5 ngày.",
            "doi_tra": "Đổi trả trong 7 ngày nếu lỗi NSX, còn tem mác.",
            "thanh_toan": "COD hoặc chuyển khoản.",
        },
    }
    return {"shop": shop, "products": products}
