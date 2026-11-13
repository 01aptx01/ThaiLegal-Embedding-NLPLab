# เริ่มใช้งาน

## เตรียมเครื่อง

- Python 3.10 ขึ้นไป และ `pip`
- Node.js 22 พร้อม npm สำหรับ build หน้าเว็บ
- อินเทอร์เน็ตสำหรับดาวน์โหลดชุดข้อมูลและโมเดลจาก Hugging Face ครั้งแรก
- CPU ใช้กับ Binary BoW, TF-IDF และ Word2Vec ได้ หากจะสร้างดัชนี BGE-M3 ครบทั้งคลังหรือเปิด Typhoon ใช้งานบ่อย แนะนำเครื่องที่ PyTorch มองเห็น CUDA

ถ้ายังไม่มีโค้ดในเครื่อง ให้ clone repo แล้วเปิด terminal ในโฟลเดอร์โปรเจกต์

```bash
git clone https://github.com/01aptx01/ThaiLegal-Embedding-NLPLab.git
cd ThaiLegal-Embedding-NLPLab
```

หากกำลังลองโค้ดจาก Pull Request ให้ checkout branch ของ PR ก่อนติดตั้ง

สร้างสภาพแวดล้อมแยกสำหรับ Python บน macOS หรือ Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

บน Windows PowerShell ใช้

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## เปิดหน้าเว็บบนเครื่อง

ติดตั้งแพ็กเกจพร้อมส่วนเสริมสำหรับเว็บ แล้ว build ไฟล์หน้าเว็บ

```bash
python -m pip install -e ".[web]"
cd frontend
npm ci
npm run build
cd ..
```

สร้างดัชนีบน CPU และเปิดเซิร์ฟเวอร์

```bash
thai-legal-rag-web build-index --no-dense --device cpu
thai-legal-rag-web serve --retrieval-only --device cpu
```

เปิด [http://127.0.0.1:8000](http://127.0.0.1:8000) แล้วลองถามว่า “บุคคลบรรลุนิติภาวะเมื่ออายุเท่าใด” เลือก TF-IDF และกดค้นหลักฐาน จากนั้นเปิดตัวบทในผลค้น โหมดนี้ใช้สามวิธีและปิดการสร้างคำตอบ ดัชนีอยู่ใน `inference_index/` หากจะเปิดใหม่ ให้รันเฉพาะ `serve` จากโฟลเดอร์เดิม

## เปิดทั้งสี่วิธีและสร้างคำตอบ

สร้างดัชนีครบในอีกโฟลเดอร์หนึ่ง แล้วเริ่มเว็บโดยเปิด Typhoon

```bash
thai-legal-rag-web build-index --output inference_index/full --device cuda
thai-legal-rag-web serve --index inference_index/full --device cuda
```

ถ้าต้องการสร้างคำตอบด้วย Typhoon แต่จะใช้เฉพาะสามวิธีค้นเดิม ให้ชี้ `--index` ไปยังดัชนีที่สร้างด้วย `--no-dense` และเอา `--retrieval-only` ออกจากคำสั่ง `serve` ลดจำนวน token ที่อนุญาตให้ตอบได้ด้วย `--max-new-tokens` ซึ่งรับค่า 1–256

## รันการทดลองจาก Terminal

ถ้าใช้เฉพาะ Notebook หรือคำสั่งทดลอง ให้ติดตั้งด้วย `python -m pip install -e .` แล้วรันคำสั่งนี้

```bash
thai-legal-rag run --mode pilot --retrieval-only
```

คำสั่งทดลองยังสร้างเวกเตอร์ BGE-M3 ให้ครบทั้งคลัง แม้เลือกคำถาม pilot น้อยหรือเปิด `--retrieval-only` ซึ่งข้ามเฉพาะการสร้างคำตอบ ถ้าต้องการลองบน CPU แบบสั้น ให้ใช้เส้นทางเปิดเว็บด้านบน ดูขั้นตอนทดสอบและรูปแบบผลที่ [การทดลองและการประเมินผล](evaluation.md)

## ปัญหาที่พบบ่อย

| อาการ | วิธีตรวจ |
| --- | --- |
| Terminal หา `thai-legal-rag-web` ไม่เจอ | เปิด virtual environment ให้ตรงกับที่ติดตั้ง `".[web]"` และรันคำสั่งจากราก repo |
| หา `frontend/dist/index.html` ไม่เจอ | รัน `npm ci` และ `npm run build` ใน `frontend/` |
| โปรแกรมแจ้งว่าโฟลเดอร์ดัชนีมีอยู่แล้ว | ใช้ดัชนีเดิมกับ `serve` หรือระบุโฟลเดอร์ใหม่ด้วย `build-index --output` |
| หน้าเว็บไม่มี BGE-M3 | ตรวจว่า `--index` ชี้ไปยังดัชนีที่สร้างโดยเปิด BGE-M3 |
| หน่วยความจำไม่พอตอนสร้างเวกเตอร์ | ลด `--batch-size` สำหรับเว็บ หรือ `--bge-batch-size` สำหรับการทดลอง |
| โฟลเดอร์ดาวน์โหลดเต็ม | ตั้งตัวแปร `HF_HOME` เป็นตำแหน่งบนดิสก์ที่มีพื้นที่ ก่อนรันคำสั่ง |
| CUDA ใช้งานไม่ได้ | ตรวจด้วย `python -c "import torch; print(torch.cuda.is_available())"` หากได้ `False` ให้เริ่มจากเว็บแบบ CPU ด้านบน |
| Evaluate ไม่มีตาราง | เปิด `serve` ด้วย `--evaluation-run` ที่ชี้ไปยังโฟลเดอร์ที่มี `retrieval_summary.csv` |
