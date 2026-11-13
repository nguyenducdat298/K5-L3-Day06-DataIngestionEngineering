# Báo Cáo Thí Nghiệm Day 06 — Data Ingestion Engineering

**Học viên:** Nguyễn Đức Đạt  
**MSSV:** 20210001  
**Lớp:** L3  
**Corpus:** NLP Foundation Papers — *Attention Is All You Need* (15 trang) & *BERT: Pre-training of Deep Bidirectional Transformers* (16 trang). **Tổng cộng: 31 trang nội dung** (đáp ứng điều kiện ≥30 trang).  
**Repository:** https://github.com/nguyenducdat298/K5-L3-Day06-DataIngestionEngineering  

---

## 1. Thiết Kế Pipeline & Lựa Chọn Dữ Liệu
- **Tài liệu nguồn:**
  - `arxiv:1706.03762`: *Attention Is All You Need* (15 trang PDF)
  - `arxiv:1810.04805`: *BERT: Pre-training of Deep Bidirectional Transformers* (16 trang PDF)
- **Chuẩn hóa (Normalization):** Khử de-hyphenation (ghép lại từ bị bẻ dòng), chuẩn hóa Unicode NFC, loại bỏ header/footer rác và chuẩn hóa khoảng trắng.
- **Mô hình Embedding & Lưu trữ:** Sử dụng `text-embedding-3-small` (1536 chiều) qua API và lưu trữ trong **PostgreSQL 16 + pgvector** với toán tử khoảng cách cosine `<=>`. Toàn bộ quá trình chạy trên **Kaggle Cloud GPU**.

---

## 2. So Sánh 3 Chiến Lược Dữ Liệu

| Chiến lược | Mô tả kỹ thuật | Số lượng Chunk | Thời gian Embed |
| :--- | :--- | :---: | :---: |
| **Strategy A (Fixed-size)** | Cửa sổ trượt cố định 500 ký tự, overlap 100 ký tự | 218 | 14.80s |
| **Strategy B (Paragraph-aware)** | Phân đoạn theo ranh giới đoạn văn (max 800 ký tự) | 136 | 9.25s |
| **Strategy C (Metadata Filter)** | Kế thừa Strategy B + lọc metadata trước truy vấn (`document_id`) | 136 | 9.25s |

---

## 3. Kết Quả Benchmark (20 Câu Hỏi: 15 Dev / 5 Test)

| Chiến lược | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 | Overall Hit@3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A (Fixed 500/100)** | 10/15 (66.7%) | 13/15 (86.7%) | 4/5 (80.0%) | 4/5 (80.0%) | **85.0%** |
| **Strategy B (Paragraph)** | 12/15 (80.0%) | 14/15 (93.3%) | 4/5 (80.0%) | 5/5 (100.0%) | **95.0%** |
| **Strategy C (Meta Filter)** | 13/15 (86.7%) | 15/15 (100.0%) | 4/5 (80.0%) | 5/5 (100.0%) | **100.0%** |

---

## 4. Đánh Giá Theo 5 Tiêu Chí (BENCHMARK.md)

1. **Độ chính xác truy xuất:** Strategy C đạt **100% Hit@3** tổng thể và 86.7% Hit@1, vượt trội so với Strategy A (85.0% Hit@3) nhờ triệt tiêu hoàn toàn khả năng nhầm lẫn giữa 2 tài liệu cùng chủ đề Transformer.
2. **Tính mạch lạc của chunk:** Strategy B và C giữ trọn vẹn ngữ nghĩa từng đoạn văn và công thức toán học, không bị xé nhỏ giữa chừng như Strategy A.
3. **Tính hữu dụng của metadata:** Thuộc tính `document_id`, `page` và `source` được lưu trữ dưới dạng JSONB và cột có index trong PostgreSQL, giúp câu hỏi có điều kiện lọc chính xác tuyệt đối.
4. **Chất lượng thông tin nền (Grounding Quality):** Top-3 của Strategy B/C luôn bao quát đủ tiền đề và định nghĩa (ví dụ: công thức Scaled Dot-Product, các tham số L/H/A của BERT).
5. **Tác động của chiến lược dữ liệu:** Chuyển từ Fixed sang Paragraph giúp giảm **37.6% số chunk** (từ 218 xuống 136), tiết kiệm thời gian và chi phí embedding tương ứng, trong khi độ chính xác tăng 10-15%.

---

## 5. Kết Luận
Strategy C (Phân đoạn theo đoạn văn kết hợp bộ lọc Metadata) là cấu hình tối ưu nhất cho bài toán RAG kỹ thuật.
