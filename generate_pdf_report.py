import os
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

base_dir = r"C:\Users\KHAZANATIL ILMI\.gemini\antigravity-ide\brain\e2fb7767-87f6-488f-babd-929e8280c742\.user_uploaded"
pdf_filename = r"C:\Users\KHAZANATIL ILMI\.gemini\antigravity-ide\scratch\handson1_mlflow\Laporan_Praktikum_MLflow_HandsOn1.pdf"

image_files = [
    ("media_1791663593698.png", "Gambar 1. Dashboard Runs Eksperimen MLflow (breast-cancer-clf)"),
    ("media_1791663650507.png", "Gambar 2. Overview & Metrics Model Baseline (rf-baseline)"),
    ("media_1791663683022.png", "Gambar 3. Parameters & Tags Model Baseline (rf-baseline)"),
    ("media_1791663703090.png", "Gambar 4. Visualisasi Grafik Metrics (val_auc & val_f1)"),
    ("media_1791663784446.png", "Gambar 5. Pelacakan Deep Learning per-Epoch PyTorch (resnet18-lr0.0003)"),
    ("media_1791663995166.png", "Gambar 6. Nested Runs Optuna Hyperparameter Tuning (optuna-sweep-01)"),
    ("media_1791663947488.png", "Gambar 7. Model Registry & Alias Champion (@champion)")
]

doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=A4,
    rightMargin=30, leftMargin=30,
    topMargin=30, bottomMargin=30
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=16,
    leading=20,
    textColor=colors.HexColor('#1A365D'),
    alignment=1,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=10,
    leading=13,
    textColor=colors.HexColor('#4A5568'),
    alignment=1,
    spaceAfter=10
)

section_style = ParagraphStyle(
    'SectionHeading',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=11.5,
    leading=15,
    textColor=colors.HexColor('#2B6CB0'),
    spaceBefore=8,
    spaceAfter=4
)

caption_style = ParagraphStyle(
    'ImgCaption',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor('#2D3748'),
    spaceBefore=3,
    spaceAfter=10
)

body_style = ParagraphStyle(
    'BodyTextCustom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#1A202C'),
    spaceAfter=4
)

story = []

# Title Header
story.append(Paragraph("<b>LAPORAN PRAKTIKUM MLOps</b>", title_style))
story.append(Paragraph("<b>Hands-on 1: Experiment Tracking & Model Registry dengan MLflow</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor('#2B6CB0'), spaceAfter=8))

# Meta Table
meta_data = [
    [Paragraph("<b>Nama Mahasiswa:</b> KHAZANATIL ILMI", body_style), Paragraph("<b>Mata Kuliah:</b> MLOps (Minggu 6)", body_style)],
    [Paragraph("<b>Topik:</b> Hands-on 1 (MLflow)", body_style), Paragraph("<b>Tanggal:</b> 11 Oktober 2026", body_style)]
]
t_meta = Table(meta_data, colWidths=[270, 260])
t_meta.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
    ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#E2E8F0')),
    ('PADDING', (0,0), (-1,-1), 5),
]))
story.append(t_meta)
story.append(Spacer(1, 8))

# Summary
story.append(Paragraph("<b>1. Ringkasan Singkat Praktikum</b>", section_style))
story.append(Paragraph(
    "Praktikum ini mengimplementasikan pelacakan eksperimen (<i>Experiment Tracking</i>) dan pengelolaan versi model (<i>Model Registry</i>) "
    "menggunakan <b>MLflow</b>. Seluruh tahap dari logging dasar scikit-learn, pelacakan per-epoch PyTorch, autologging, "
    "hyperparameter tuning Optuna (nested runs), hingga registrasi model dengan alias <code>@champion</code> berhasil dieksekusi tanpa mengubah nilai output.", body_style
))
story.append(Spacer(1, 6))

def load_img(fname, target_width=530, max_h=260):
    ipath = os.path.join(base_dir, fname)
    img = Image.open(ipath)
    w, h = img.size
    aspect = h / float(w)
    th = target_width * aspect
    if th > max_h:
        th = max_h
        target_width = th / aspect
    return RLImage(ipath, width=target_width, height=th)

# Section 2: Screenshots & Explanations
story.append(Paragraph("<b>2. Hasil Eksekusi & Screenshot Eksperimen</b>", section_style))

# SS 1
story.append(load_img(image_files[0][0]))
story.append(Paragraph(f"<b>{image_files[0][1]}</b>: Menampilkan seluruh daftar eksekusi run pada eksperimen <code>breast-cancer-clf</code> di MLflow UI.", caption_style))

# SS 2
story.append(load_img(image_files[1][0]))
story.append(Paragraph(f"<b>{image_files[1][1]}</b>: Ringkasan metrik model baseline <code>rf-baseline</code> (val_f1 = 0.9655, val_auc = 0.9940).", caption_style))

# PageBreak for neat reading
story.append(PageBreak())

# SS 3
story.append(load_img(image_files[2][0]))
story.append(Paragraph(f"<b>{image_files[2][1]}</b>: Pelacakan parameter (n_estimators=200, max_depth=8) dan tags (stage: baseline, author: luky).", caption_style))

# SS 4
story.append(load_img(image_files[3][0]))
story.append(Paragraph(f"<b>{image_files[3][1]}</b>: Visualisasi bar chart hasil evaluasi metrik val_auc dan val_f1.", caption_style))

# SS 5
story.append(load_img(image_files[4][0]))
story.append(Paragraph(f"<b>{image_files[4][1]}</b>: Kurva pelacakan per-epoch PyTorch untuk loss, f1-score, dan learning rate.", caption_style))

# PageBreak
story.append(PageBreak())

# SS 6 (Optuna)
story.append(load_img(image_files[5][0]))
story.append(Paragraph(f"<b>{image_files[5][1]}</b>: Struktur nested runs Optuna (1 parent run membawahi 10 child trials).", caption_style))

# SS 7 (Registry)
story.append(load_img(image_files[6][0]))
story.append(Paragraph(f"<b>{image_files[6][1]}</b>: Model versi 1 terdaftar pada Model Registry dan diberi alias <code>@champion</code>.", caption_style))

# Kesimpulan
story.append(Paragraph("<b>3. Kesimpulan Ringkas</b>", section_style))
story.append(Paragraph(
    "Seluruh 6 langkah Hands-on 1 MLflow telah berjalan 100% sukses. MLflow terbukti mempermudah reproduksi eksperimen, "
    "pelacakan hyperparameter/metrik, serta tata kelola model siap produksi.", body_style
))

doc.build(story)
print("Updated PDF generated successfully!")
