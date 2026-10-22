# Literature GAP Matrix — RAG for Medication Review with Incomplete Information

Tugas kuliah Generative AI. Terdapat 13 sumber ilmiah yang diidentifikasi menggunakan sumber daring penerbit/PubMed/ACL; `papers.csv` menyimpan **parafrasa ringkasan**, bukan PDF teks penuh. Gap merupakan **interpretasi calon gap**, bukan klaim novelty yang definitif.

## Alur pipeline
`papers.csv` → normalisasi metadata → TF-IDF retrieval (cosine) → ambil top-k bukti untuk research questions → generasi matriks (`offline` terstruktur; atau `groq` LLM) → CSV gap matrix + jejak retrieval.

## Menjalankan tanpa API
```bash
python rag_gap_pipeline.py --mode offline
```
Mode offline memiliki retrieval nyata, tetapi tahap generasi **deterministik/template**, bukan LLM. Ini adalah baseline yang dapat dijalankan tanpa token atau internet.

## Menjalankan pipeline RAG + LLM Groq
```bash
pip install groq
# Linux/macOS: export GROQ_API_KEY="..."
# Windows PowerShell: $env:GROQ_API_KEY="..."
python rag_gap_pipeline.py --mode groq
```
Model default `openai/gpt-oss-20b`, dapat diubah via `GROQ_MODEL`; periksa ketersediaan model di akun Groq. Jangan simpan API key dalam repositori.

## Output
`output/literature_gap_matrix.csv`, `output/retrieval_trace.csv`, dan `output/run_metadata.json`.

## Batasan ilmiah
- Retrieval di sini menggunakan ringkasan sumber beranotasi, bukan indeks PDF full-text. Tidak tepat mengklaim otomatis membaca keseluruhan makalah.
- Metadata dan gap awal dikurasi manual berdasarkan sumber terverifikasi; model tidak boleh mengarang temuan yang tidak ada di sumber.
- Kolom `candidate_gap` adalah hipotesis untuk dibahas dan diverifikasi dari full-text; gap global memerlukan pencarian literatur sistematis yang lebih luas.
- Tidak menggunakan informasi pasien asli atau dokumen riset internal yang bersifat rahasia.
- Pengunggahan ke GitHub dan LMS perlu dilakukan oleh pemilik akun (atau konektor berizin).
