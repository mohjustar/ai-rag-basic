import os
import weaviate
import google.generativeai as genai
from flask import Flask, request, render_template, flash, redirect, url_for
from weaviate.classes.init import Auth
# Import yang benar untuk konfigurasi properti di v4
from weaviate.classes.config import Configure, Property, DataType

# --- Inisialisasi Aplikasi dan Klien ---

app = Flask(__name__)
# Diperlukan untuk menampilkan 'flash messages'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- Konfigurasi Klien API ---
try:
    # Konfigurasi Gemini API dari environment variable
    google_api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
    # Model untuk menjawab pertanyaan
    generative_model = genai.GenerativeModel('gemini-2.0-flash')
    # Model untuk membuat embeddings (vektor)
    embedding_model = 'models/embedding-001'
    print("Model Gemini berhasil dikonfigurasi.")
except KeyError:
    generative_model = None
    embedding_model = None
    print("Kesalahan: GOOGLE_API_KEY tidak ditemukan. Harap atur environment variable.")
except Exception as e:
    generative_model = None
    embedding_model = None
    print(f"Terjadi kesalahan saat konfigurasi Gemini: {e}")

# --- KODE YANG DIPERBARUI UNTUK KONEKSI WEAVIATE CLOUD (WCS) v4 ---
client = None # Inisialisasi client sebagai None
try:
    # Ambil kredensial dari environment variables
    wcs_url = os.environ["WEAVIATE_URL"]
    wcs_api_key = os.environ["WEAVIATE_API_KEY"]

    # Gunakan metode connect_to_weaviate_cloud yang terbaru
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcs_url,
        auth_credentials=Auth.api_key(wcs_api_key),
    )

    # Periksa apakah Weaviate siap
    if client.is_ready():
        print("Klien Weaviate Cloud (WCS) berhasil terhubung (v4).")
    else:
        print("Klien Weaviate tidak siap.")
        client = None

except KeyError as e:
    print(f"Kesalahan: Variabel lingkungan {e} tidak ditemukan. Pastikan WEAVIATE_URL dan WEAVIATE_API_KEY sudah diatur.")
except Exception as e:
    print(f"Terjadi kesalahan saat menghubungkan ke Weaviate Cloud: {e}")


# --- Skema Weaviate ---

def setup_weaviate_schema():
    """Mendefinisikan dan membuat skema di Weaviate jika belum ada."""
    if not client:
        return

    class_name = "TextChunk"
    # Cek apakah class sudah ada
    if not client.collections.exists(class_name):
        print(f"Skema '{class_name}' tidak ditemukan. Membuat skema baru...")
        
        client.collections.create(
            name=class_name,
            description="Potongan teks dari dokumen yang diunggah",
            # PERBAIKAN: Hapus vectorizer_config agar Weaviate menerima vektor buatan sendiri
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source_document", data_type=DataType.TEXT),
            ]
        )
        print(f"Skema '{class_name}' berhasil dibuat untuk mode 'Bring Your Own Vector'.")
    else:
        print(f"Skema '{class_name}' sudah ada.")

# Panggil fungsi setup saat aplikasi pertama kali dijalankan
with app.app_context():
    # Hanya jalankan setup jika klien berhasil terhubung
    if client:
        setup_weaviate_schema()

# --- Fungsi Bantuan (Helper) ---

def chunk_text(text, chunk_size=256, overlap=30):
    """Membagi teks menjadi beberapa bagian yang saling tumpang tindih."""
    words = text.split()
    if not words:
        return []
    chunks = []
    # Iterasi dengan langkah (chunk_size - overlap) untuk membuat tumpang tindih
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Rute Flask ---

@app.route('/')
def index():
    """Menampilkan halaman utama."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Menangani unggahan file, memproses, dan menyimpannya di Weaviate."""
    if not client or not generative_model:
        flash("Sistem tidak terkonfigurasi dengan benar. Periksa koneksi ke API.", "error")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('Tidak ada bagian file yang terdeteksi.', 'error')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih untuk diunggah.', 'error')
        return redirect(request.url)

    if file and file.filename.endswith('.txt'):
        try:
            filename = file.filename
            text_content = file.read().decode('utf-8')
            chunks = chunk_text(text_content)

            text_chunks_collection = client.collections.get("TextChunk")
            with text_chunks_collection.batch.dynamic() as batch:
                for chunk in chunks:
                    # PERBAIKAN: Buat embedding untuk setiap chunk menggunakan Gemini
                    embedding = genai.embed_content(
                        model=embedding_model,
                        content=chunk,
                        task_type="RETRIEVAL_DOCUMENT"
                    )["embedding"]

                    properties = {
                        "content": chunk,
                        "source_document": filename
                    }
                    # Tambahkan objek beserta vektornya
                    batch.add_object(properties=properties, vector=embedding)

            flash(f"File '{filename}' berhasil diproses dan diindeks menjadi {len(chunks)} bagian.", "success")
        except Exception as e:
            flash(f"Terjadi kesalahan saat memproses file: {e}", "error")
    else:
        flash("Harap unggah file dengan format .txt", "error")

    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    """Mengambil konteks dari Weaviate dan menghasilkan jawaban menggunakan Gemini."""
    question = request.form.get('question')
    if not question:
        flash("Silakan ajukan pertanyaan.", "error")
        return redirect(url_for('index'))

    if not client or not generative_model:
        flash("Sistem tidak terkonfigurasi dengan benar. Periksa koneksi ke API.", "error")
        return redirect(url_for('index'))

    try:
        # PERBAIKAN: Buat embedding untuk pertanyaan pengguna
        question_embedding = genai.embed_content(
            model=embedding_model,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

        text_chunks_collection = client.collections.get("TextChunk")
        # PERBAIKAN: Gunakan near_vector untuk mencari berdasarkan vektor
        response = text_chunks_collection.query.near_vector(
            near_vector=question_embedding,
            limit=3
        )
        
        context_chunks_obj = response.objects
        if not context_chunks_obj:
            answer = "Maaf, saya tidak dapat menemukan informasi yang relevan di dokumen yang tersedia untuk menjawab pertanyaan Anda."
            return render_template('index.html', question=question, answer=answer, context=[])

        context_for_template = [obj.properties for obj in context_chunks_obj]
        context_for_prompt = "\n---\n".join([item['content'] for item in context_for_template])

        # 2. AUGMENTATION: Buat prompt untuk Gemini dengan konteks yang ditemukan
        prompt = f"""
        Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan dari sebuah dokumen.
        Jawablah pertanyaan berikut secara ringkas dan jelas hanya berdasarkan konteks di bawah ini.
        Jika jawaban tidak ada dalam konteks, katakan "Berdasarkan informasi yang ada, saya tidak dapat menjawab pertanyaan tersebut."

        Konteks:
        {context_for_prompt}

        Pertanyaan: {question}

        Jawaban:
        """

        # 3. GENERATION: Hasilkan jawaban menggunakan Gemini
        generated_response = generative_model.generate_content(prompt)
        answer = generated_response.text

        return render_template('index.html', question=question, answer=answer, context=context_for_template)

    except Exception as e:
        flash(f"Terjadi kesalahan saat memproses pertanyaan: {e}", "error")
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='0.0.0.0', port=5001, debug=True)

    # PERBAIKAN: Menutup koneksi saat aplikasi dihentikan
    if client:
        client.close()
