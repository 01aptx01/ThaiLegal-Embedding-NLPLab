# Thai Legal RAG

ชุดทดลองเปรียบเทียบการค้นคืนตัวบทกฎหมายไทย 4 วิธี และประเมินการสร้างคำตอบพร้อม citation จากหลักฐานที่ค้นคืนได้ โค้ด Python แบบ package แยกขั้นตอนทำงานออกจาก [notebook ต้นฉบับ](notebooks/Thai_Legal_RAG_Clean.ipynb) เพื่อให้รันซ้ำ ตรวจสอบผล และนำเข้าคะแนน human review ได้สะดวก

> ผลลัพธ์เป็นงานทดลองทางวิชาการ ไม่ใช่คำแนะนำทางกฎหมาย

## เริ่มต้นใช้งาน

ต้องใช้ Python 3.10 ขึ้นไป แนะนำให้สร้าง virtual environment แยกไว้ก่อน โมเดลและชุดข้อมูลจะดาวน์โหลดจาก Hugging Face ในการรันครั้งแรก และอาจต้องใช้พื้นที่ดิสก์กับ RAM/GPU มาก

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

รัน pilot แบบ retrieval-only ซึ่งยังสร้าง embeddings ให้ครบทั้งคลังมาตรา

```bash
thai-legal-rag run --mode pilot --retrieval-only
```

รัน pilot พร้อม generation ค่าเริ่มต้นสูงสุด 40 คำถามสำหรับ retrieval และ 4 คำถามสำหรับ generation

```bash
thai-legal-rag run --mode pilot
```

รัน test benchmark และสุ่มคำถามสำหรับ generation โดยกระจายตามกฎหมายสูงสุด 200 ข้อ

```bash
thai-legal-rag run --mode test
```

ตัวอย่างการระบุอุปกรณ์และโฟลเดอร์ผลลัพธ์

```bash
thai-legal-rag run --mode pilot --device cpu --output-dir ./rag_results
```

`--device auto` ใช้ CUDA เมื่อพร้อม มิฉะนั้นใช้ CPU; กำหนดเป็น `cpu` หรือ `cuda` ได้เอง `--bge-batch-size` ใช้ลด batch เมื่อหน่วยความจำไม่พอ ค่าปริยายคือ 8

## ขั้นตอนและโครงสร้างโค้ด

ลำดับด้านล่างอ้างอิงเลขเซลล์แบบเริ่มนับจาก 1 ใน notebook ต้นฉบับ:

| Notebook | งาน | โมดูล Python |
| --- | --- | --- |
| 3–4 | ติดตั้ง dependencies, กำหนดค่าเริ่มต้น และเริ่ม manifest | `pyproject.toml`, `config.py`, `cli.py`, `pipeline.py` |
| 6 | โหลด datasets และบันทึก revision ที่ใช้ | `data.py`, `pipeline.py` |
| 8–9 | สร้างรหัส statute, จับคู่ gold, ตรวจ duplicate/ambiguous และคัดคำถาม | `data.py` |
| 11 | เลือกชุดคำถาม generation แบบแบ่งชั้นตามกฎหมาย | `data.py`, `pipeline.py` |
| 13 | ตรวจความยาว corpus ตาม tokenizer ของ BGE-M3 | `retrieval.py` |
| 15–16 | สร้าง ranking ด้วย Binary BoW, TF-IDF, Word2Vec และ BGE-M3 | `retrieval.py` |
| 18–19 | คำนวณ retrieval metrics และสรุปผล | `retrieval.py` |
| 21 | สร้างคำตอบจากหลักฐานที่ใส่ใน prompt พร้อมตรวจ citation | `generation.py` |
| 23 | สร้างสรุป generation และ review template | `generation.py` |
| 25 | ตรวจและสรุปคะแนน human review | `review.py` |
| 27 | รวมผลเป็น ZIP | `pipeline.py` |

คำสั่งหลักติดตั้งเป็น `thai-legal-rag` และเรียกผ่าน `thai_legal_rag.cli:main` ส่วน notebook เก็บ workflow ต้นฉบับไว้เพื่อเทียบกับการทำงานแบบ package

## คำสั่ง

```text
thai-legal-rag run [--mode {pilot,test}] [--retrieval-only]
                   [--output-dir PATH] [--device {auto,cpu,cuda}]
                   [--seed INT] [--pilot-questions INT]
                   [--pilot-generation-questions INT]
                   [--test-generation-questions INT]
                   [--corpus-revision REF] [--pilot-revision REF]
                   [--test-revision REF] [--bge-revision REF]
                   [--generator-revision REF] [--bge-batch-size INT]

thai-legal-rag review RUN_DIR [--scores PATH]
```

ค่า revision เป็น commit hash หรือ reference ที่ Hugging Face เข้าใจได้ หากไม่ระบุ โปรแกรมจะ resolve revision ปัจจุบันและบันทึกค่าที่ใช้ใน manifest ของ run การระบุ revision ทั้งห้าช่วยให้เรียกชุดข้อมูล corpus, pilot, test และโมเดล BGE-M3 กับ generator รุ่นเดิมได้ เมื่อต้องเทียบรอบทดลอง ให้ใช้ `--seed`, revisions, รุ่น Python/dependencies และ device เดิม กำหนด `PYTHONHASHSEED` ก่อนเปิด Python เพื่อควบคุมการสุ่ม hash ระหว่าง process

## ผลลัพธ์

แต่ละ run สร้างโฟลเดอร์ `rag_results/<mode>_.../` เมื่อรันสำเร็จจะได้ ZIP อยู่ข้างกัน

- `manifest.json`, `run_config.json` — revisions, รุ่น dependencies, การตั้งค่า และ ID ของคำถามที่ใช้
- `statute_ids.csv`, `duplicate_keys.jsonl`, `question_audit.csv`, `mapping_audit.csv` — ตารางรหัสกฎหมายและ audit การเตรียมข้อมูล
- `corpus_lengths.csv` — ความยาว corpus เมื่อ tokenize ด้วย BGE-M3 และจำนวนที่เกินขีดจำกัด
- `retrieval_rows.jsonl`, `retrieval_summary.csv`, `retrieval_on_generation_subset.csv` — rankings, metrics และเวลา
- `generation_rows.jsonl`, `generation_summary.csv`, `evidence_stages.csv` — คำตอบ, citation metrics และจุดที่หลักฐานขาดหาย
- `review_template.csv` — แบบฟอร์มให้ผู้ประเมินกรอกคะแนน
- `review_scored.csv`, `human_summary.csv`, `human_evidence_stages.csv` — คะแนนและสรุปหลังนำเข้าการประเมิน

หากเปิด generation ให้คัดลอก `review_template.csv` เป็น `review_scored.csv` ในโฟลเดอร์ run แล้วกรอกคอลัมน์ `coverage` (0–2), `contradiction` (0–1) และ `citation_support` (0–1) คง `review_id` เดิมไว้ จากนั้นนำเข้าคะแนนด้วยคำสั่งนี้

```bash
thai-legal-rag review rag_results/<ชื่อโฟลเดอร์-run>
```

ถ้าเก็บไฟล์คะแนนไว้นอกโฟลเดอร์ run ให้ระบุ path

```bash
thai-legal-rag review rag_results/<ชื่อโฟลเดอร์-run> --scores /path/to/review_scored.csv
```

Review จะรับเฉพาะ `review_id` จาก run นั้น และสรุปเฉพาะคำถามที่มีคะแนนครบทั้ง 4 วิธี โปรแกรมเขียน `human_summary.csv` กับ `human_evidence_stages.csv` ใหม่; เมื่อมีคำถามที่ครบอย่างน้อยหนึ่งข้อ จะ refresh ZIP ของ run ด้วย

## ขอบเขตและข้อควรตีความ

- ค่าเริ่มต้น pilot ใช้ได้สูงสุด 40 คำถาม; generation ใช้ 4 คำถามใน pilot และสูงสุด 200 ใน test Retrieval ใช้คำถามที่จับคู่ gold ได้ครบ ส่วน generation เลือกจากคำถามเหล่านั้นที่มี reference answer
- คำถามที่ว่าง ไม่มี gold, จับคู่ตัวบทไม่ได้หรือกำกวม และ pilot ที่ซ้ำกับ test จะไม่เข้า evaluation จำนวน eligible จริงอาจเปลี่ยนตาม revision ของ Hugging Face; ตรวจได้ใน `question_audit.csv` และ `mapping_audit.csv`
- Retrieval ใช้ exact cosine ranking กับ Binary BoW, TF-IDF, Word2Vec ที่ฝึกจาก corpus ของ run และ BGE-M3; R1–R3 ใช้ Thai word tokenization เดียวกัน ผลจึงขึ้นกับ corpus, preprocessing, model revision และ seed
- ใช้หนึ่งมาตราเป็นหนึ่งหน่วยค้นคืน ทั้งสี่วิธีใช้ชุดมาตราเดียวกัน วัด Recall@1/5/10, Hit@5, nDCG@5/10 และ MRR@10 แบบ macro average รายคำถาม
- BGE-M3 ถูกจำกัดความยาว input ตามการตั้งค่า ส่วน generation จำกัด context ด้วย token budget และตัด statute ออกเป็นทั้งส่วนเมื่อใส่ prompt ไม่พอดี ตัวบทที่ยาวกว่า BGE limit อาจถูกตัดโดย tokenizer ของ encoder
- Citation metrics ตรวจรูปแบบรหัสและการอ้างรหัสที่อยู่ใน prompt แต่ไม่ยืนยันความถูกต้องทางกฎหมายหรือว่าข้อความรองรับข้อสรุปจริง ควรใช้ human review ประกอบ และอย่าใช้ผลลัพธ์แทนการตรวจโดยผู้เชี่ยวชาญ
- งานเต็มอาจใช้เวลานานและกินหน่วยความจำ โดยเฉพาะ BGE-M3 และ generator การรันบน CPU ทำได้แต่ช้ากว่ามาก; ถ้า memory ไม่พอให้ลด `--bge-batch-size` และถ้าพื้นที่ดาวน์โหลดโมเดลไม่พอให้ตั้งตัวแปรสภาพแวดล้อม `HF_HOME` ไปยังดิสก์ที่มีพื้นที่

## ตรวจสอบชุดทดสอบ

```bash
python -m unittest discover -s tests -v
```
