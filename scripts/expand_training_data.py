import argparse
import random
from typing import Dict, List

import pandas as pd


def _build_templates() -> Dict[str, List[str]]:
    return {
        "Lịch học": [
            "Xem thời khóa biểu {term} ở đâu",
            "Lịch học môn {subj} cập nhật khi nào",
            "Mình bị trùng lịch học {term} thì xử lý thế nào",
            "Đổi lớp học phần môn {subj} được không",
            "Lịch học tuần này có thay đổi không",
            "Xin link xem lịch học {term}",
            "Lịch học buổi {slot} bắt đầu lúc mấy giờ",
            "Môn {subj} học ở cơ sở {campus} hay cơ sở {campus2}",
            "Lịch học bị đổi phòng thì xem ở đâu",
            "Cách xem lịch học theo mã sinh viên",
        ],
        "Điểm số": [
            "Khi nào có điểm môn {subj}",
            "Xem điểm {term} ở đâu",
            "Điểm quá trình môn {subj} đã lên chưa",
            "Điểm thi cuối kỳ môn {subj} khi nào công bố",
            "Mình muốn phúc khảo điểm môn {subj} thì làm sao",
            "Bảng điểm tổng kết {term} tải ở đâu",
            "Điểm rèn luyện {term} xem ở đâu",
            "Điểm bị sai thì liên hệ {dept} hay {dept2}",
            "Điểm học lại môn {subj} có tính vào GPA không",
            "Điểm thi lại bao giờ cập nhật",
            "Điểm {exam} môn {subj} khi nào có",
            "Điểm {exam} hiển thị sai thì xử lý thế nào",
            "Vì sao điểm môn {subj} chưa cập nhật",
            "Mình muốn xin bảng điểm {lang} thì nộp ở đâu",
            "Điểm trung bình {term} tính như thế nào",
            "GPA {term} xem ở đâu",
            "Điểm chữ {grade} quy đổi ra thang 10 thế nào",
            "Điểm thiếu cột {component} thì liên hệ ai",
            "Thời hạn phúc khảo điểm {exam} là khi nào",
            "Kết quả phúc khảo điểm môn {subj} bao giờ có",
        ],
        "Học phí": [
            "Hạn nộp học phí {term} là ngày nào",
            "Xem công nợ học phí {term} ở đâu",
            "Nộp học phí online qua {channel} được không",
            "Biên lai học phí {term} lấy ở đâu",
            "Học phí môn {subj} tính như thế nào",
            "Nếu nộp học phí trễ {term} thì có bị phạt không",
            "Đã chuyển khoản học phí nhưng chưa thấy cập nhật",
            "Học phí có thể đóng qua app ngân hàng không",
            "Miễn giảm học phí cần giấy tờ gì",
            "Giờ làm việc của {dept} thu học phí là mấy giờ",
            "Học phí {term} đóng ở đâu cho nhanh",
            "Em muốn xin xác nhận đã đóng học phí {term} ở đâu",
            "Nộp học phí qua {bank} có được không",
            "Học phí {term} có thể đóng nhiều lần không",
        ],
        "Tài khoản": [
            "Quên mật khẩu tài khoản {channel} thì làm sao",
            "Không đăng nhập được {channel} do sai mật khẩu",
            "Cách đổi mật khẩu tài khoản sinh viên",
            "Tài khoản bị khóa thì mở lại như thế nào",
            "Email sinh viên không nhận được mã xác thực",
            "Cập nhật số điện thoại trong hồ sơ tài khoản ở đâu",
            "Portal báo tài khoản không tồn tại thì xử lý sao",
            "Không nhận được mã OTP khi đăng nhập",
            "LMS báo lỗi đăng nhập {code} thì làm sao",
            "Mình nghi bị lộ mật khẩu thì cần làm gì",
            "Không nhận được email reset mật khẩu thì xử lý sao",
            "Đổi email liên hệ trong hồ sơ tài khoản ở đâu",
            "Tài khoản {channel} bị khoá do đăng nhập nhiều lần",
            "Cách bật xác thực 2 lớp cho tài khoản portal",
        ],
        "Thủ tục hành chính": [
            "Xin giấy xác nhận sinh viên ở {dept} hay {dept2}",
            "Thủ tục xin nghỉ học tạm thời gồm những gì",
            "Cách làm đơn bảo lưu kết quả học tập",
            "Làm thẻ sinh viên bị mất cần giấy tờ gì",
            "Xin bảng điểm tiếng Anh nộp hồ sơ ở đâu",
            "Thủ tục chuyển ngành có được không",
            "Xin giấy giới thiệu thực tập nộp ở {dept}",
            "Thủ tục xin thôi học gồm những bước nào",
            "Xin sao y bảng điểm cần chuẩn bị gì",
            "Xin xác nhận không nợ học phí ở đâu",
            "Xin giấy xác nhận đang học để làm {purpose} ở đâu",
            "Nộp đơn {form} cần những giấy tờ gì",
            "Thủ tục xin cấp lại thẻ sinh viên mất bao lâu",
            "Xin giấy giới thiệu {purpose} cần làm thế nào",
        ],
        "Khác": [
            "Giờ làm việc của {dept} là mấy giờ",
            "Liên hệ {dept} qua số điện thoại nào",
            "Trường có hỗ trợ wifi ký túc xá không",
            "Hỏi về hoạt động câu lạc bộ sinh viên",
            "Lịch nghỉ lễ {term} xem ở đâu",
            "Có tổ chức workshop về {subj} không",
            "Thư viện mở cửa lúc mấy giờ",
            "Căn tin mở mấy giờ",
            "Bãi giữ xe gần cổng nào",
            "Trong trường có cây ATM không",
            "Chỗ gửi xe gần toà {building} ở đâu",
            "Khu ký túc xá {gender} ở đâu",
            "Trường có hỗ trợ in ấn ở {dept} không",
            "Hôm nay có sự kiện gì ở trường không",
        ],
    }


def _render(
    tpl: str,
    *,
    subjects: List[str],
    terms: List[str],
    channels: List[str],
    depts: List[str],
    slots: List[str],
    campuses: List[str],
    codes: List[str],
    prefixes: List[str],
    suffixes: List[str],
    exams: List[str],
    langs: List[str],
    grades: List[str],
    components: List[str],
    banks: List[str],
    purposes: List[str],
    forms: List[str],
    buildings: List[str],
    genders: List[str],
) -> str:
    campus = random.choice(campuses)
    campus2 = random.choice([c for c in campuses if c != campus] or campuses)
    dept = random.choice(depts)
    dept2 = random.choice([d for d in depts if d != dept] or depts)
    base = tpl.format(
        subj=random.choice(subjects),
        term=random.choice(terms),
        channel=random.choice(channels),
        dept=dept,
        dept2=dept2,
        slot=random.choice(slots),
        campus=campus,
        campus2=campus2,
        code=random.choice(codes),
        exam=random.choice(exams),
        lang=random.choice(langs),
        grade=random.choice(grades),
        component=random.choice(components),
        bank=random.choice(banks),
        purpose=random.choice(purposes),
        form=random.choice(forms),
        building=random.choice(buildings),
        gender=random.choice(genders),
    ).strip()

    text = f"{random.choice(prefixes)}{base}{random.choice(suffixes)}".strip()
    # chuẩn hoá khoảng trắng
    return " ".join(text.split())


def expand_csv(csv_path: str, add: int, seed: int) -> None:
    random.seed(seed)
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")

    existing = set(df["text"].astype(str).str.strip().tolist())

    labels = [
        "Lịch học",
        "Điểm số",
        "Học phí",
        "Tài khoản",
        "Thủ tục hành chính",
        "Khác",
    ]

    subjects = [
        "AI",
        "Toán rời rạc",
        "Xác suất",
        "Lập trình Python",
        "Cơ sở dữ liệu",
        "Mạng máy tính",
        "NLP",
        "Hệ điều hành",
        "An toàn thông tin",
    ]
    terms = ["học kỳ 1", "học kỳ 2", "học kỳ này", "kỳ hè", "kỳ phụ"]
    channels = [
        "cổng thông tin",
        "website trường",
        "phòng đào tạo",
        "khoa",
        "giáo vụ",
        "trợ lý học vụ",
    ]
    depts = [
        "phòng đào tạo",
        "phòng công tác sinh viên",
        "phòng tài chính",
        "thư viện",
        "khoa",
    ]
    slots = ["sáng", "chiều", "tối"]
    campuses = ["1", "2", "A", "B"]
    codes = ["403", "500", "401"]
    prefixes = ["", "Cho em hỏi ", "Mình muốn hỏi ", "Xin hỏi ", "Anh/chị cho hỏi "]
    suffixes = [
        "",
        " trên portal",
        " trên cổng thông tin",
        " vậy",
        " được không",
        " giúp mình với",
        " với ạ",
    ]
    exams = ["giữa kỳ", "cuối kỳ", "quá trình", "thi lại", "cải thiện"]
    langs = ["tiếng Việt", "tiếng Anh"]
    grades = ["A", "B", "C", "D", "F", "I"]
    components = ["quá trình", "giữa kỳ", "cuối kỳ", "chuyên cần"]
    banks = ["Vietcombank", "BIDV", "VietinBank", "MB Bank", "ACB"]
    purposes = ["vay vốn", "xin việc", "thực tập", "học bổng", "làm thêm"]
    forms = ["bảo lưu", "nghỉ học", "chuyển ngành", "rút học phần"]
    buildings = ["A1", "A2", "B1", "C", "D"]
    genders = ["nam", "nữ"]

    templates = _build_templates()

    per_label = {lab: add // len(labels) for lab in labels}
    for lab in labels[: add % len(labels)]:
        per_label[lab] += 1

    new_rows = []
    for label in labels:
        remaining = per_label[label]
        tries = 0
        label_templates = templates[label]
        while remaining > 0 and tries < 500000:
            tries += 1
            text = _render(
                random.choice(label_templates),
                subjects=subjects,
                terms=terms,
                channels=channels,
                depts=depts,
                slots=slots,
                campuses=campuses,
                codes=codes,
                prefixes=prefixes,
                suffixes=suffixes,
                exams=exams,
                langs=langs,
                grades=grades,
                components=components,
                banks=banks,
                purposes=purposes,
                forms=forms,
                buildings=buildings,
                genders=genders,
            )
            # nhỏ hoá xác suất trùng lặp
            if random.random() < 0.25:
                text = text + " ạ"

            if text in existing:
                continue

            existing.add(text)
            new_rows.append({"text": text, "label": label})
            remaining -= 1

        if remaining > 0:
            raise RuntimeError(
                f"Could not generate enough unique rows for label '{label}'. Missing {remaining}."
            )

    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(csv_path, index=False)

    print(f"Added {len(new_rows)} rows. Total rows = {len(out)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/training_data.csv")
    parser.add_argument("--add", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.add <= 0:
        raise SystemExit("--add must be > 0")

    expand_csv(args.csv, add=args.add, seed=args.seed)


if __name__ == "__main__":
    main()
