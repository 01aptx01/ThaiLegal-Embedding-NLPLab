# ที่มาของภาพใน README

ภาพหน้าเว็บจับด้วย Chromium/Playwright จาก frontend และ FastAPI ของ repo ที่รันจริง ไม่มีการแทน API response หรือใส่คำตอบจำลองในหน้าเว็บ ส่วนภาพผลคำสั่งและสัญญาอนุญาตเป็นภาพของข้อความจริงที่จัดรูปให้อ่านง่าย ไม่ใช่ภาพจากโปรแกรม Terminal

## สภาพแวดล้อมที่ใช้

- โค้ดอ้างอิง commit `392109054adae83ae0f42a2d2dd6a11401bc3f16`
- Python 3.12, Node.js 24.22.1, CPU, `--retrieval-only`, ดัชนี `--no-dense`
- คลัง CCL 5,127 มาตรา รุ่น `069e774f015a50e801dbe2055a29be580c54a83c`
- ใช้งาน R1 Binary BoW, R2 TF-IDF และ R3 Word2Vec; ไม่ได้โหลด BGE-M3 หรือ Typhoon
- ภาพหน้าเว็บกว้าง 1,440 พิกเซล รองรับฟอนต์ Noto Sans Thai

## ภาพแต่ละส่วน

| ไฟล์ | ที่มา |
| --- | --- |
| `overview.png` | Playground หลัง API และดัชนีพร้อม ก่อนส่งคำถาม |
| `playground.png` | ส่งคำถาม “บุคคลบรรลุนิติภาวะเมื่ออายุเท่าใด” ให้ R2 TF-IDF แล้วเปิดหลักฐาน |
| `compare.png` | ส่งคำถาม “ผู้เยาว์ทำนิติกรรมโดยไม่ได้รับความยินยอมจากผู้แทนโดยชอบธรรม มีผลอย่างไร” ให้ R1–R3 |
| `evaluate.png` | ตารางจากไฟล์ `evaluation/retrieval_summary.csv` ที่คำนวณจริงตามขั้นตอนด้านล่าง |
| `history.png` | ประวัติจากการค้นและเปรียบเทียบสองครั้งข้างต้นใน browser session เดียวกัน |
| `trace.png` | เปิด Trace ของผลค้น R2 จากคำถามใน Playground |
| `getting-started.png` | stdout ของ frontend build และบรรทัดสำคัญจากการสร้างดัชนีจริง |
| `configuration.png` | stdout ของ `thai-legal-rag-web serve --help` |
| `contributing.png` | บางรายชื่อจาก unittest พร้อมสรุปจริง 32 tests ผ่าน และ stdout ของ frontend build |
| `license.png` | ข้อความจากไฟล์ `LICENSE` ของ repo |

## ผลที่ใช้ใน Evaluate

รอบนี้สร้างขึ้นเพื่อประกอบภาพ README โดยใช้ดัชนีจริงและคำถาม pilot 40 ข้อ seed 42 เป็นผลค้นสามวิธีบน CPU ไม่ใช่รอบ benchmark ครบสี่วิธีของคำสั่ง `thai-legal-rag run` และไม่มีผลการสร้างคำตอบหรือคะแนน human review

1. โหลดทั้งสามชุดข้อมูลด้วย `load_sources` ใน `data.py` และบันทึก revision ลง [manifest](evaluation/manifest.json)
2. ใช้ `prepare_data(raw, "pilot", 40, 42, audit_dir)` เพื่อจับคู่มาตรา กรองคำถามซ้ำกับ test และเลือก pilot ด้วยกติกาของโปรเจกต์
3. โหลด `InferenceIndex` ที่สร้างบนคลังรุ่นเดียวกัน ค้นแต่ละคำถามด้วย R1–R3 ครั้งละ 10 มาตรา
4. ใช้ `score_ranking` ใน `retrieval.py` คำนวณ Recall@5, Hit@5, nDCG@5 และ RR@10 แล้วเฉลี่ยข้ามคำถาม โดย MRR@10 คือค่าเฉลี่ย RR@10
5. บันทึก [ผลรายข้อ](evaluation/retrieval_rows.jsonl) และ [ตารางสรุป](evaluation/retrieval_summary.csv) ตารางปัดทศนิยมสามตำแหน่ง ส่วน `query_ms` วัดเวลาของ `index.search` รวมการตัดคำ เวลาขึ้นอยู่กับเครื่องและแต่ละรอบที่รัน

หากต้องการเปิดตารางที่ใช้จับภาพ หลังติดตั้งและสร้างดัชนีตาม README ให้ใช้

```bash
thai-legal-rag-web serve --retrieval-only --device cpu \
  --evaluation-run docs/screenshots/evaluation
```

เปิด `http://127.0.0.1:8000/#evaluate` การค้นคำถามใหม่ยังใช้ดัชนีบนเครื่อง ส่วนตาราง Evaluate อ่านผลที่บันทึกไว้ในโฟลเดอร์นี้

## การตรวจสอบ

- `python -m unittest discover -s tests -v` ผ่าน 32 ข้อ
- `npm --prefix frontend run build` ผ่าน
- ทดสอบผ่าน browser ตั้งแต่ส่งคำถาม → `/api/infer` → ดัชนีจริง → แสดงหลักฐานและ Trace → Compare → Evaluate → History
- ไม่พบ JavaScript page error หรือ API response ที่เป็น HTTP error ระหว่างจับภาพ

เวลาค้นในภาพและในตารางเป็นค่าจากคนละคำขอ จึงไม่จำเป็นต้องเท่ากัน ผลค้นในภาพเป็นข้อมูลจากคลังสำหรับการทดลอง ให้ตรวจตัวบทฉบับปัจจุบันกับแหล่งทางการก่อนนำไปอ้างอิง
