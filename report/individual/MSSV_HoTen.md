# Báo Cáo Thí Nghiệm Day 06 — Data Ingestion Engineering

**Học viên:** Nguyễn Đức Đạt  
**MSSV:** 2A202603544  
**Lớp:** L3  
**Corpus:** NLP Foundation Papers — *Attention Is All You Need* (`arxiv:1706.03762`, 15 trang) & *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (`arxiv:1810.04805`, 16 trang). **Tổng cộng: 31 trang nội dung** (đáp ứng yêu cầu ≥ 30 trang).  
**Repository:** https://github.com/nguyenducdat298/K5-L3-Day06-DataIngestionEngineering  
**Kaggle Kernel (v3 — `KernelWorkerStatus.COMPLETE`):** https://www.kaggle.com/code/tommydatlc/day06-rag-ingestion-benchmark  

---

## 1. Thiết Kế Pipeline & Chuẩn Hóa Dữ Liệu
- **Trích xuất PDF:** Sử dụng `pypdf` đọc từng trang vật lý (`1..N`) và giữ nguyên metadata `document_id`, `source`, `page`, `total_pages`. Tổng thời gian ingest 2 tài liệu (31 trang) trên Kaggle là **2.351s**.
- **Chuẩn hóa văn bản (`normalize_text`):**
  1. Chuẩn hóa Unicode **NFKC** để xử lý các ký tự ghép ligature thường gặp trong PDF (như `ﬁ` → `fi`, `ﬂ` → `fl` trong bài báo BERT).
  2. Khử ngắt từ qua dòng (de-hyphenation): `([A-Za-z0-9])-\s*\n\s*([A-Za-z0-9])` → ghép liền từ bị tách.
  3. Khôi phục ranh giới đoạn văn và tiêu đề mục (vì `pypdf` thường chỉ xuất `\n` đơn thay vì `\n\n`) dựa trên regex nhận diện heading (`3.1 ...`, `Task #1`, `Table ...`, `Figure ...`, `Appendix ...`) và dấu kết thúc câu ở dòng ngắn.
- **Embedding & Vector Store (100% chạy trên Kaggle Cloud GPU):**
  - Model: **`google/embeddinggemma-2`** (**768 chiều**, load từ `/kaggle/input/models/google/embeddinggemma-2/transformers/embeddinggemma-2/1` qua `sentence-transformers`).
  - Vector Database: **PostgreSQL + pgvector v0.7.0**, lưu bảng `rag_chunks` với khóa `UNIQUE (strategy, chunk_id)`, cột `metadata JSONB`, index trên `(strategy, document_id, page)`, truy vấn cosine distance `<=>`.

---

## 2. So Sánh 3 Chiến Lược Chunking & Retrieval

| Chiến lược | Mô tả kỹ thuật | Số Chunk | Thời gian Chunk | Thời gian Embed | Thời gian Upsert | Thời gian Query (20 câu) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A (`A_fixed`)** | Cửa sổ cố định 500 ký tự, overlap 100 | 262 | 0.0010s | 11.330s | 0.616s | 0.1392s |
| **Strategy B (`B_paragraph`)** | Gom đoạn văn (max 800 ký tự), tách câu khi vượt ngưỡng | 185 | 0.0021s | 7.911s | 0.438s | 0.1210s |
| **Strategy C (`C_meta_filter`)** | Chunk của B + lọc SQL `WHERE` theo metadata (`document_id`) | 185 | 0.0021s | 7.911s (tái dùng vector B) | 0.413s | 0.1215s |

---

## 3. Kết Quả Benchmark Chính Thức (20 Câu Hỏi: 15 Dev / 5 Test)

Dữ liệu từ `starter/results/benchmark_results.json` và `artifacts/experiment_log.csv` (chạy thực tế trên Kaggle GPU v3):

| Chiến lược | Overall Hit@1 | Overall Hit@3 | Dev Hit@1 (15) | Dev Hit@3 (15) | Test Hit@1 (5) | Test Hit@3 (5) | Filtered Hit@3 (4 câu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A (`A_fixed`)** | 90.0% (18/20) | 95.0% (19/20) | 86.7% (13/15) | 93.3% (14/15) | 100.0% (5/5) | 100.0% (5/5) | 75.0% (3/4) |
| **Strategy B (`B_paragraph`)** | 90.0% (18/20) | **100.0% (20/20)** | **93.3% (14/15)** | **100.0% (15/15)** | 80.0% (4/5) | 100.0% (5/5) | **100.0% (4/4)** |
| **Strategy C (`C_meta_filter`)** | 90.0% (18/20) | **100.0% (20/20)** | **93.3% (14/15)** | **100.0% (15/15)** | 80.0% (4/5) | 100.0% (5/5) | **100.0% (4/4)** |

---

## 4. Đánh Giá Theo 5 Tiêu Chí (`BENCHMARK.md`)

### 4.1. Độ chính xác truy xuất (Retrieval Accuracy — Hit@1 & Evidence Hit@3)
- **Strategy B và C đạt 100.0% Hit@3 (20/20)** và **90.0% Hit@1 (18/20)** trên toàn bộ benchmark (15/15 dev, 5/5 test).
- **Strategy A (`A_fixed`)** chỉ đạt **95.0% Hit@3 (19/20)** và **93.3% Dev Hit@3 (14/15)** do trượt hoàn toàn câu **`q14`** (cả 3 hạng đầu không tìm được trang 13 của BERT chứa tham số pre-training batch size 256 & 1,000,000 steps). Ngoài ra câu `q07` ở Strategy A bị đẩy xuống Rank 3, còn Strategy B/C đều truy xuất đúng ở Rank 1.

### 4.2. Tính mạch lạc của Chunk (Coherence — Kiểm tra mẫu 10 chunk thực tế)
So sánh mẫu trích từ `coherence_samples` trong `benchmark_results.json`:
- **Strategy A (Fixed 500/100)** cắt ngang giữa câu/từ ở biên cửa sổ:
  - `transformer-2017_p5_f5` (500 ký tự): kết thúc bị cụt ở `"use different parameters from layer to layer. Another wa"`.
  - `bert-2018_p6_f6` (500 ký tự): mở đầu bị cắt nửa họ `"rkar et al., 2016)"` và kết thúc cụt `"with the question us"`.
  - `bert-2018_p12_f1` (500 ký tự): mở đầu cụt `"olume 1: Long Papers)"`, kết thúc cụt `"G"`.
- **Strategy B/C (Paragraph <= 800)** giữ trọn ranh giới mục, đoạn văn và tài liệu tham khảo:
  - `transformer-2017_p5_para4` (525 ký tự): bắt đầu trọn mục `"3.4 Embeddings and Softmax..."` và kết thúc hoàn chỉnh `"by √dmodel. 5"`.
  - `bert-2018_p5_para4` (617 ký tự): giữ trọn đoạn mô tả đầu ra fine-tuning đến hết câu `"found in Appendix A.5."`.

### 4.3. Tính hữu dụng của Metadata (Metadata Usefulness — 4 câu có filter: `q13, q14, q15` [dev] & `q19` [test])
- Cả 4 câu hỏi lọc metadata (`{"document_id": "transformer-2017"}` hoặc `"bert-2018"`) đều đạt **4/4 (100%) ở Rank 1** với Strategy C (`C_meta_filter`).
- Với 4 câu hỏi này, Strategy B (không filter) cũng đạt Rank 1 do tên tài liệu (`Transformer paper`, `BERT paper`) đã xuất hiện trong câu hỏi và EmbeddingGemma-2 tách biệt tốt không gian ngữ nghĩa hai bài báo. Tuy vậy, trong thực tế production nhiều tài liệu hơn (nhiều năm/phiên bản trùng thuật ngữ), bộ lọc `WHERE document_id = %s` của Strategy C đảm bảo **loại trừ 100% nhiễu chéo tài liệu** ở tầng SQL trước khi tính cosine similarity.

### 4.4. Chất lượng thông tin nền tập Test (Grounding Quality — 5/5 câu test của Strategy C)
Kiểm tra trực tiếp top-1 chunk trả về cho 5 câu test (`q16–q20`):
1. `q16` (Rank 1, score `0.8262`, `transformer-2017_p5_para0`): Chứa đoạn phân tích Multi-Head Attention ở trang 5 (`d_k = d_v = d_model/h = 64`).
2. `q17` (Rank 1, score `0.8222`, `transformer-2017_p5_para3`): Giữ trọn mục `3.3 Position-wise Feed-Forward Networks` cùng phương trình ReLU `FFN(x) = max(0, xW1 + b1)W2 + b2`.
3. `q18` (Rank 2 — Hit@3, gold `bert-2018_p4_para1`, score `0.7795`): Chứa chính xác câu `"We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary."` (Rank 1 trả về chú thích `Figure 2` trên trang 5 giải thích Input Embeddings).
4. `q19` (Rank 1, score `0.8379`, `bert-2018_p7_para1`, filtered `bert-2018`): Trả về trọn `Table 2: SQuAD 1.1 results` với số liệu `BERT_LARGE (Single) 84.1 EM / 90.9 F1`.
5. `q20` (Rank 1, score `0.8734`, `bert-2018_p4_para1`): Giữ trọn phần mô tả cách đóng gói cặp câu qua token `[SEP]` và segment embedding A/B.

### 4.5. Tác động của chiến lược dữ liệu & Phân tích lỗi (Error Analysis)
- **Tối ưu tài nguyên & chất lượng đồng thời:** Chuyển từ Fixed-size (262 chunks) sang Paragraph-aware (185 chunks) **giảm 29.4% số chunk** (và **giảm 30.2% thời gian embedding** từ 11.33s xuống 7.911s), đồng thời **nâng Overall Hit@3 từ 95.0% lên 100.0%**.
- **Phân tích ca lỗi tiêu biểu:**
  - `q14` (miss ở Strategy A): Cửa sổ cố định 500 ký tự chia nhỏ Appendix A.2 (trang 13 BERT), làm ngữ cảnh câu `"batch size of 256 sequences... for 1,000,000 steps"` bị loãng; pgvector kéo về `bert-2018_p12_f6` (tiêu đề Appendix A.1, score `0.8043`) và `bert-2018_p14_f4` (bảng so sánh GPT). Strategy B giữ trọn đoạn Appendix A.2 nên đẩy đúng chunk lên Rank 1.
  - `q08` (Hit@1 = False nhưng Hit@3 = True ở Strategy B/C): Rank 1 trả về `bert-2018_p12_para5` (score `0.804`) có câu `"the masked LM only make predictions on 15% of tokens in each batch"`, vốn **thực chất vẫn trả lời đúng 15%**, nhưng nằm ở trang 12 (Ablation) thay vì trang 4 (định nghĩa Task #1 — xếp Rank 2, score `0.7966`). Điều này gợi ý mở rộng gold page list cho các ý xuất hiện ở nhiều phần trong tài liệu.

---

## 5. Kết Luận
Cấu hình **Strategy C (Paragraph-aware chunking + SQL Metadata Filter với `pgvector` và `google/embeddinggemma-2`)** đạt cân bằng tốt nhất: giảm gần 30% dung lượng vector index và thời gian embedding so với chia cố định, đạt **100.0% Hit@3** trên cả tập dev và test, đồng thời đảm bảo độ tin cậy cô lập nguồn dữ liệu qua JSONB/SQL filter.
