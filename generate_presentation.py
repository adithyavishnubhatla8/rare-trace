import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path="RARETRACE_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Professional Clinical Color Palette
    C_NAVY_DARK = RGBColor(15, 23, 42)      # #0F172A
    C_NAVY_LIGHT = RGBColor(30, 41, 59)     # #1E293B
    C_TEAL = RGBColor(13, 148, 136)         # #0D9488
    C_TEAL_LIGHT = RGBColor(20, 184, 166)   # #14B8A6
    C_BG_LIGHT = RGBColor(248, 250, 252)    # #F8FAFC
    C_WHITE = RGBColor(255, 255, 255)
    C_TEXT_DARK = RGBColor(15, 23, 42)
    C_TEXT_MUTED = RGBColor(100, 116, 139)  # #64748B
    C_CARD_BORDER = RGBColor(226, 232, 240) # #E2E8F0
    C_RED = RGBColor(225, 29, 72)           # #E11D48
    C_BLUE_ACCENT = RGBColor(2, 132, 199)   # #0284C7

    def add_header(slide, title_text, category="RARETRACE CLINICAL INTELLIGENCE"):
        # Top banner background shape
        header_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        header_bg.fill.solid()
        header_bg.fill.fore_color.rgb = C_NAVY_DARK
        header_bg.line.fill.background()

        # Accent line under header
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.15), Inches(13.333), Inches(0.06))
        line.fill.solid()
        line.fill.fore_color.rgb = C_TEAL
        line.line.fill.background()

        # Category / breadcrumb
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.3))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_TEAL_LIGHT

        # Slide Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.5), Inches(0.6))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = C_WHITE

        # Footer
        add_footer(slide)

    def add_footer(slide):
        footer_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.02))
        footer_line.fill.solid()
        footer_line.fill.fore_color.rgb = C_CARD_BORDER
        footer_line.line.fill.background()

        tb = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.35))
        p = tb.text_frame.paragraphs[0]
        p.text = "RARETRACE — DBSCAN-Based Anomaly Detection for Rare Diseases | B.Tech Capstone 2026–2027"
        p.font.size = Pt(9)
        p.font.color.rgb = C_TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=C_WHITE, border_color=C_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    # ==========================================
    # SLIDE 1: Title Slide (Dark Theme)
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY_DARK
    bg1.line.fill.background()

    # Teal accent bar
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.15))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = C_TEAL
    top_bar.line.fill.background()

    # Badge
    b1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.2), Inches(3.2), Inches(0.45))
    b1.fill.solid()
    b1.fill.fore_color.rgb = RGBColor(19, 78, 74)
    b1.line.color.rgb = C_TEAL
    p_b1 = b1.text_frame.paragraphs[0]
    p_b1.text = "CLINICAL INTELLIGENCE PLATFORM"
    p_b1.font.size = Pt(11)
    p_b1.font.bold = True
    p_b1.font.color.rgb = C_TEAL_LIGHT
    p_b1.alignment = PP_ALIGN.CENTER

    # Project Title
    t1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.85), Inches(11.0), Inches(1.6))
    tf1 = t1.text_frame
    tf1.word_wrap = True
    p1_1 = tf1.paragraphs[0]
    p1_1.text = "RARETRACE"
    p1_1.font.size = Pt(44)
    p1_1.font.bold = True
    p1_1.font.color.rgb = C_WHITE

    p1_2 = tf1.add_paragraph()
    p1_2.text = "Rare-Disease Case Identification Using DBSCAN-Based Anomaly Detection"
    p1_2.font.size = Pt(22)
    p1_2.font.color.rgb = C_TEAL_LIGHT
    p1_2.space_before = Pt(8)

    # Subtitle / Abstract statement
    sub1 = s1.shapes.add_textbox(Inches(1.2), Inches(3.6), Inches(10.5), Inches(1.0))
    p_sub = sub1.text_frame.paragraphs[0]
    p_sub.text = "An unsupervised machine learning framework integrating density-based spatial clustering, adaptive k-NN knee optimization, clinical decision support (CDS), and Supabase PostgreSQL audit trails."
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)
    sub1.text_frame.word_wrap = True

    # Card with Authors & Guide Details
    c_meta = add_card(s1, Inches(1.2), Inches(4.8), Inches(10.9), Inches(1.8), bg_color=C_NAVY_LIGHT, border_color=RGBColor(51, 65, 85))
    tb_meta = s1.shapes.add_textbox(Inches(1.5), Inches(4.95), Inches(10.3), Inches(1.5))
    tf_meta = tb_meta.text_frame
    
    pm1 = tf_meta.paragraphs[0]
    pm1.text = "CAPSTONE PROJECT DETAILS (2026–2027)"
    pm1.font.size = Pt(11)
    pm1.font.bold = True
    pm1.font.color.rgb = C_TEAL_LIGHT

    pm2 = tf_meta.add_paragraph()
    pm2.text = "• Student Developers: Adithya Vishnubhatla & Pavan Kumar"
    pm2.font.size = Pt(13)
    pm2.font.bold = True
    pm2.font.color.rgb = C_WHITE
    pm2.space_before = Pt(4)

    pm3 = tf_meta.add_paragraph()
    pm3.text = "• Faculty Guide: I V Sai Lakshmi Haritha | Department of Computer Science & Engineering"
    pm3.font.size = Pt(12)
    pm3.font.color.rgb = RGBColor(203, 213, 225)
    pm3.space_before = Pt(3)

    pm4 = tf_meta.add_paragraph()
    pm4.text = "• Core Architecture: Unsupervised DBSCAN ML • Supabase PostgreSQL • Vercel Serverless"
    pm4.font.size = Pt(11)
    pm4.font.color.rgb = C_TEAL_LIGHT
    pm4.space_before = Pt(3)

    # ==========================================
    # SLIDE 2: Clinical Motivation & The Problem
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Clinical Background: The Diagnostic Odyssey")

    # Card 1: The Global Crisis
    add_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), bg_color=C_WHITE)
    tb2_1 = s2.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf2_1 = tb2_1.text_frame
    tf2_1.word_wrap = True
    p = tf2_1.paragraphs[0]
    p.text = "The Rare-Disease Challenge"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    bullets1 = [
        ("300+ Million Patients", "Over 300 million individuals globally live with one of ~7,000 recognized rare conditions."),
        ("4.8 to 7.3 Years Delay", "Patients visit an average of 8 physicians and receive 2-3 misdiagnoses before correct identification."),
        ("Diagnostic Odyssey", "Long delays lead to irreversible organ damage, disease progression, and massive emotional/financial strain."),
        ("Clinical Heterogeneity", "Symptoms overlap broadly with prevalent chronic conditions, confounding routine clinical reviews.")
    ]
    for title, desc in bullets1:
        p_t = tf2_1.add_paragraph()
        p_t.text = f"• {title}: "
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_RED
        p_t.space_before = Pt(8)
        
        p_d = tf2_1.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Card 2: Failure of Traditional AI
    add_card(s2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2), bg_color=C_BG_LIGHT)
    tb2_2 = s2.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf2_2 = tb2_2.text_frame
    tf2_2.word_wrap = True
    p2 = tf2_2.paragraphs[0]
    p2.text = "Why Supervised ML Fails"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = C_NAVY_DARK

    bullets2 = [
        ("Extreme Class Imbalance", "Rare cases comprise < 0.1% of clinical cohorts. Standard classifiers collapse into trivial majority-class predictions."),
        ("Absence of Ground Truth Labels", "Novel or ultra-rare variants lack verified positive training sets in historical Electronic Health Records (EHR)."),
        ("Overfitting on Artifacts", "Supervised models overfit to spurious patterns rather than genuine phenotypic anomalies."),
        ("The Solution: Unsupervised Outlier Mining", "Treat rare diseases not as a classification task, but as mathematical spatial anomalies in multidimensional clinical space.")
    ]
    for title, desc in bullets2:
        p_t = tf2_2.add_paragraph()
        p_t.text = f"• {title}: "
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(8)

        p_d = tf2_2.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_MUTED

    # ==========================================
    # SLIDE 3: Problem Statement & Objectives
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Project Objectives & Technical Scope")

    cards3 = [
        ("1. Unsupervised Discovery", "Detect anomalous clinical records from unlabelled heterogeneous patient populations without requiring historical disease labels or ground truth.", C_TEAL),
        ("2. Density-Based Clustering", "Implement DBSCAN to separate typical disease clusters from non-conforming isolated noise points (label = -1) exhibiting rare pathology signatures.", C_BLUE_ACCENT),
        ("3. Adaptive Parameter Tuning", "Dynamically calculate optimal eps radii using k-nearest neighbor (k-NN) distance graphs and mathematical knee/elbow detection.", C_NAVY_DARK),
        ("4. Explainable Clinical AI (XAI)", "Derive population Z-score feature attributions for every flagged candidate so clinicians understand precisely why an anomaly was triggered.", C_RED),
        ("5. Clinical Decision Support", "Generate structured specialist referrals, confirmatory biomarker tests, and evidence-based diagnostic pathways for attending physicians.", C_TEAL),
        ("6. Cloud-Native Audit & Security", "Persist de-identified analysis metadata in Supabase PostgreSQL while strictly maintaining zero-PII transmission for HIPAA compliance.", C_BLUE_ACCENT),
    ]

    for idx, (title, desc, color) in enumerate(cards3):
        col = idx % 3
        row = idx // 3
        x = Inches(0.8 + col * 4.0)
        y = Inches(1.5 + row * 2.65)
        
        c = add_card(s3, x, y, Inches(3.7), Inches(2.4), bg_color=C_WHITE)
        
        # Color bar top of card
        c_bar = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(3.7), Inches(0.08))
        c_bar.fill.solid()
        c_bar.fill.fore_color.rgb = color
        c_bar.line.fill.background()

        tb = s3.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(3.3), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = color

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_DARK
        p_d.space_before = Pt(6)

    # ==========================================
    # SLIDE 4: End-to-End System Workflow
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "RARETRACE Architectural Pipeline & Workflow")

    workflow_steps = [
        ("Step 1", "CSV Ingestion", "Upload clinical dataset (EHR, vitals, lab biomarkers)."),
        ("Step 2", "Validation", "Schema check, missingness scan, type enforcement."),
        ("Step 3", "Preprocessing", "Clinical imputation & Robust/Standard scaling."),
        ("Step 4", "Feature Eng.", "Biomarker interaction & risk ratio synthesis."),
        ("Step 5", "DBSCAN Mining", "Adaptive k-NN knee optimization & clustering."),
        ("Step 6", "Anomaly Scoring", "Distance-to-core metric & candidate extraction."),
        ("Step 7", "XAI Attribution", "2D PCA coordinates & Z-score feature ranking."),
        ("Step 8", "CDS Engine", "Specialist mapping & clinical recommendations."),
        ("Step 9", "Supabase Sync", "Audit metadata written to Supabase PostgreSQL.")
    ]

    for idx, (step_num, step_title, step_desc) in enumerate(workflow_steps):
        col = idx % 3
        row = idx // 3
        x = Inches(0.8 + col * 4.0)
        y = Inches(1.5 + row * 1.75)

        card = add_card(s4, x, y, Inches(3.7), Inches(1.55), bg_color=C_WHITE)
        tb = s4.shapes.add_textbox(x + Inches(0.2), y + Inches(0.12), Inches(3.3), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = f"{step_num}: {step_title}"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = C_TEAL

        p2 = tf.add_paragraph()
        p2.text = step_desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = C_TEXT_MUTED
        p2.space_before = Pt(4)

    # Bottom summary bar
    sum_box = add_card(s4, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.75), bg_color=C_NAVY_DARK, border_color=None)
    tb_sum = s4.shapes.add_textbox(Inches(1.0), Inches(6.1), Inches(11.3), Inches(0.55))
    p_sum = tb_sum.text_frame.paragraphs[0]
    p_sum.text = "Workflow Guarantee: Zero PII transmission • Ephemeral serverless execution • Full reproducibility across multi-tenant cohorts"
    p_sum.font.size = Pt(11.5)
    p_sum.font.bold = True
    p_sum.font.color.rgb = C_TEAL_LIGHT
    p_sum.alignment = PP_ALIGN.CENTER

    # ==========================================
    # SLIDE 5: Data Preprocessing & Feature Engineering
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Clinical Data Preprocessing & Feature Engineering")

    # Column 1: Preprocessing
    add_card(s5, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), bg_color=C_WHITE)
    tb5_1 = s5.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True
    p = tf5_1.paragraphs[0]
    p.text = "1. Clinical Preprocessing Pipeline"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    prep_points = [
        ("Dynamic Column Detection", "Automatically identifies clinical biomarkers (WBC, Platelets, Creatinine, ALT, Blood Pressure, Glucose) and separates metadata from numerical tensors."),
        ("Medical Imputation", "Missing clinical parameters are imputed using median-based cohort values to prevent artificial outlier distortion caused by arbitrary zero-filling."),
        ("Robust Standardization", "Scales features using Z-score standardization: z = (x - μ) / σ. Prevents high-magnitude lab tests from dominating Euclidean distance calculations."),
        ("Noise Filtering", "Validates biological ranges to ensure sensor artifacts or data-entry errors are handled prior to spatial density calculations.")
    ]
    for t, d in prep_points:
        p_t = tf5_1.add_paragraph()
        p_t.text = f"• {t}"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(8)
        p_d = tf5_1.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Column 2: Feature Engineering
    add_card(s5, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2), bg_color=C_BG_LIGHT)
    tb5_2 = s5.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf5_2 = tb5_2.text_frame
    tf5_2.word_wrap = True
    p = tf5_2.paragraphs[0]
    p.text = "2. Domain-Specific Feature Engineering"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    eng_points = [
        ("Metabolic & Hematologic Ratios", "Synthesizes clinically validated diagnostic ratios (e.g., BUN/Creatinine, AST/ALT, Neutrophil-to-Lymphocyte ratio) critical in autoimmune and metabolic rare diseases."),
        ("Multisystem Composite Risk Index", "Calculates compounded physiological deviation scores across disparate organ systems (cardiac, renal, hepatic, immunological)."),
        ("Non-Linear Interaction Terms", "Captures cross-biomarker dependencies that individually appear borderline normal but jointly signal rare multi-system disease phenotypes."),
        ("Dimensionality Preparation", "Outputs high-density n-dimensional Euclidean vectors ready for spatial clustering without premature variance loss.")
    ]
    for t, d in eng_points:
        p_t = tf5_2.add_paragraph()
        p_t.text = f"• {t}"
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = C_BLUE_ACCENT
        p_t.space_before = Pt(8)
        p_d = tf5_2.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_TEXT_MUTED

    # ==========================================
    # SLIDE 6: DBSCAN Anomaly Detection Methodology
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "DBSCAN Clustering & Adaptive Parameter Optimization")

    # Left: How DBSCAN Works
    add_card(s6, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), bg_color=C_WHITE)
    tb6_1 = s6.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    p = tf6_1.paragraphs[0]
    p.text = "Core DBSCAN Principles"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    dbscan_concepts = [
        ("Density-Based Paradigm", "Unlike K-Means, DBSCAN does not assume spherical clusters or require specifying 'k' clusters in advance. It discovers arbitrary-shaped dense regions."),
        ("Core Points", "A point is a Core Point if at least 'min_samples' points fall within Euclidean radius 'eps'."),
        ("Border Points", "Points reachable from a core point within distance 'eps' but possessing fewer than 'min_samples' neighbors."),
        ("Noise / Anomaly Points (Label = -1)", "Points that are neither core nor border points. In RARETRACE, these isolated points represent candidate rare-disease phenotypes."),
        ("Continuous Anomaly Score", "Calculated as the normalized distance from the candidate to the nearest core cluster centroid.")
    ]
    for t, d in dbscan_concepts:
        p_t = tf6_1.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(6)
        p_d = tf6_1.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Right: Adaptive Knee Optimization
    add_card(s6, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2), bg_color=C_BG_LIGHT)
    tb6_2 = s6.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    p = tf6_2.paragraphs[0]
    p.text = "Adaptive k-NN Knee Optimization"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    knee_points = [
        ("The Hyperparameter Bottleneck", "Fixed 'eps' values fail across varying cohort dimensions, causing either zero outliers or 50% false positives."),
        ("k-Nearest Neighbors (k-NN) Graph", "For every patient record, calculate distance to its k-th nearest neighbor where k = 2 * features - 1."),
        ("Automated Knee / Elbow Detection", "Sort k-distances in ascending order and compute second-order numerical derivatives to detect the point of maximum curvature (the 'knee')."),
        ("Mathematical Optimum", "The knee point corresponds to the threshold where intra-cluster density transitions into sparse background noise, establishing the optimal 'eps'."),
        ("Full User Override", "Clinical researchers retain full manual slider control to tune 'eps' and 'min_samples' to adjust screening sensitivity.")
    ]
    for t, d in knee_points:
        p_t = tf6_2.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = C_BLUE_ACCENT
        p_t.space_before = Pt(6)
        p_d = tf6_2.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_MUTED

    # ==========================================
    # SLIDE 7: Explainable AI & Clinical Attribution
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Explainability: Z-Score Feature Attribution & PCA")

    # 3 Cards Layout
    c1 = add_card(s7, Inches(0.8), Inches(1.5), Inches(3.7), Inches(5.2), bg_color=C_WHITE)
    tb7_1 = s7.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(3.3), Inches(4.8))
    tf7_1 = tb7_1.text_frame
    tf7_1.word_wrap = True
    p = tf7_1.paragraphs[0]
    p.text = "1. Z-Score Deviation"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_RED

    p_body1 = tf7_1.add_paragraph()
    p_body1.text = "Formula: z = (x_patient - μ_pop) / σ_pop\n\nEvery candidate patient's biomarkers are compared directly against the reference cohort mean and standard deviation.\n\n• High Z (> +2.0): Biomarker is significantly elevated.\n• Low Z (< -2.0): Biomarker is severely suppressed.\n• Absolute Deviation Ranking: Features with |Z| >= 1.7 are highlighted as abnormal drivers of anomaly."
    p_body1.font.size = Pt(11)
    p_body1.font.color.rgb = C_TEXT_DARK
    p_body1.space_before = Pt(10)

    c2 = add_card(s7, Inches(4.8), Inches(1.5), Inches(3.7), Inches(5.2), bg_color=C_WHITE)
    tb7_2 = s7.shapes.add_textbox(Inches(5.0), Inches(1.7), Inches(3.3), Inches(4.8))
    tf7_2 = tb7_2.text_frame
    tf7_2.word_wrap = True
    p = tf7_2.paragraphs[0]
    p.text = "2. 2D PCA Projections"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEAL

    p_body2 = tf7_2.add_paragraph()
    p_body2.text = "Dimensionality Reduction:\nPrincipal Component Analysis (PCA) maps multidimensional clinical spaces down to 2 principal components (PC1, PC2).\n\n• Instant Spatial Intuition: Visualizes dense normal clusters vs. distant isolated red anomaly markers.\n• Fast Zero-Latency Rendering: Coordinates are pre-computed during pipeline execution, ensuring interactive 60 FPS dashboard exploration."
    p_body2.font.size = Pt(11)
    p_body2.font.color.rgb = C_TEXT_DARK
    p_body2.space_before = Pt(10)

    c3 = add_card(s7, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.2), bg_color=C_WHITE)
    tb7_3 = s7.shapes.add_textbox(Inches(9.0), Inches(1.7), Inches(3.3), Inches(4.8))
    tf7_3 = tb7_3.text_frame
    tf7_3.word_wrap = True
    p = tf7_3.paragraphs[0]
    p.text = "3. Patient Dossier"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE_ACCENT

    p_body3 = tf7_3.add_paragraph()
    p_body3.text = "Clinician-Facing Summary:\nNo black-box predictions. Each flagged patient receives a synthesized dossier:\n\n• Anomaly Score: 0.00 to 1.00\n• Review Priority: High, Moderate, Low\n• Key Abnormal Drivers: E.g., 'Elevated Ferritin (+3.8σ), Severe Thrombocytopenia (-2.9σ)'\n• Diagnostic Confidence: Cluster density vs. isolation distance."
    p_body3.font.size = Pt(11)
    p_body3.font.color.rgb = C_TEXT_DARK
    p_body3.space_before = Pt(10)

    # ==========================================
    # SLIDE 8: Clinical Decision Support Engine
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Clinical Decision Support (CDS) & Specialist Mapping")

    add_card(s8, Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.2), bg_color=C_WHITE)
    tb8 = s8.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "Transforming Spatial Outliers into Actionable Medical Guidance"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    cds_grid = [
        ("Evidence-Based Rule Engine", "Maps anomalous biomarker combinations to recognized clinical patterns (e.g., microangiopathic hemolytic anemia, systemic autoinflammation, lysosomal storage disorders)."),
        ("Multidisciplinary Specialist Referrals", "Recommends specific referral departments: Medical Genetics, Pediatric Immunology, Rare Hematology, Rheumatology, or Metabolic Endocrinology."),
        ("Confirmatory Diagnostic Roadmap", "Lists secondary validation tests: Whole Exome Sequencing (WES), targeted gene panels, enzyme activity assays, or high-resolution organ MRI."),
        ("Clinical Urgency Stratification", "Assigns Priority Levels: 'CRITICAL REVIEW' (immediate multi-organ risk), 'HIGH' (suspected rare variant), or 'MODERATE' (outlier monitoring)."),
        ("Clinician-in-the-Loop Governance", "All automated CDS recommendations serve strictly as diagnostic decision support; final authority remains with certified physicians.")
    ]

    for t, d in cds_grid:
        p_t = tf8.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(8)
        p_d = tf8.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_MUTED

    # ==========================================
    # SLIDE 9: Experimental Evaluation & Benchmark
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Experimental Results & Model Evaluation")

    # Table of Evaluation Metrics
    add_card(s9, Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.2), bg_color=C_WHITE)
    tb9_1 = s9.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.6), Inches(4.8))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True

    p = tf9_1.paragraphs[0]
    p.text = "Clustering Validation Metrics"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    metrics_text = [
        ("Silhouette Coefficient", "Measures intra-cluster cohesion vs. inter-cluster separation. Scores >= 0.40 indicate robust clinical cluster definitions."),
        ("Davies-Bouldin Index", "Evaluates cluster similarity ratio; lower values signify well-isolated, compact physiological clusters."),
        ("Calinski-Harabasz Index", "Variance ratio criterion confirming high between-cluster dispersion compared to within-cluster dispersion."),
        ("Noise Ratio Stability", "Consistently isolates 1.5% to 4.5% candidate cases, matching recognized rare disease prevalence distributions without excessive alarm fatigue.")
    ]
    for t, d in metrics_text:
        p_t = tf9_1.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(8)
        p_d = tf9_1.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Comparison with Isolation Forest
    add_card(s9, Inches(7.1), Inches(1.5), Inches(5.4), Inches(5.2), bg_color=C_BG_LIGHT)
    tb9_2 = s9.shapes.add_textbox(Inches(7.3), Inches(1.7), Inches(5.0), Inches(4.8))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True

    p = tf9_2.paragraphs[0]
    p.text = "DBSCAN vs. Isolation Forest Benchmark"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    bench_bullets = [
        ("Ensemble Agreement", "Cross-verified against Isolation Forest (iForest). RARETRACE tracks consensus overlap percentage between tree partition depth and density radius."),
        ("Preservation of Multi-Modal Clusters", "Isolation Forest isolates points regardless of background cluster topology. DBSCAN preserves healthy baseline clusters while detecting true density gaps."),
        ("Boundary Sensitivity", "DBSCAN avoids false-positive flagging at the periphery of dense normal clusters by enforcing density reachability."),
        ("Linear Execution Runtime", "Completes clustering on 1,000+ patient records in < 1.2 seconds, making it ideal for real-time web deployment.")
    ]
    for t, d in bench_bullets:
        p_t = tf9_2.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_BLUE_ACCENT
        p_t.space_before = Pt(8)
        p_d = tf9_2.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_TEXT_MUTED

    # ==========================================
    # SLIDE 10: Supabase PostgreSQL & Cloud Architecture
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Database Integration & Vercel Serverless Architecture")

    # 4 Tech Stack pillars
    pillars = [
        ("1. Supabase PostgreSQL", [
            "Cloud PostgreSQL database with Row Level Security (RLS).",
            "Stores pipeline metadata, DBSCAN hyperparameters, and cluster statistics.",
            "RESTful PostgREST integration via secure server-side requests.",
            "Zero PII stored remotely; complete patient privacy guaranteed."
        ], C_TEAL),
        ("2. Vercel Serverless Engine", [
            "100% serverless Python WSGI runtime via `api/index.py`.",
            "Stateless execution: ephemeral file handling in `/tmp`.",
            "Vercel Edge CDN for high-performance static asset caching.",
            "Deploys seamlessly via `vercel` and `vercel --prod`."
        ], C_BLUE_ACCENT),
        ("3. Web & Analytical Core", [
            "Flask 3.x web application with Jinja2 responsive templates.",
            "Light theme design system with deep navy typography.",
            "Dynamic dataset switching and multi-cohort workspace isolation.",
            "Integrated PDF Clinical Report generator via ReportLab."
        ], C_NAVY_DARK),
        ("4. Cross-Platform Client", [
            "Progressive Web App (PWA) with service worker offline caching.",
            "Trusted Web Activity (TWA) Android APK build.",
            "Responsive layout adapted for desktop, tablets, and mobile clinical carts."
        ], C_RED)
    ]

    for idx, (title, items, color) in enumerate(pillars):
        col = idx % 2
        row = idx // 2
        x = Inches(0.8 + col * 5.9)
        y = Inches(1.5 + row * 2.65)

        card = add_card(s10, x, y, Inches(5.6), Inches(2.45), bg_color=C_WHITE)
        
        # Color bar
        c_bar = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(5.6), Inches(0.06))
        c_bar.fill.solid()
        c_bar.fill.fore_color.rgb = color
        c_bar.line.fill.background()

        tb = s10.shapes.add_textbox(x + Inches(0.2), y + Inches(0.15), Inches(5.2), Inches(2.1))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color

        for item in items:
            p_i = tf.add_paragraph()
            p_i.text = f"• {item}"
            p_i.font.size = Pt(10.5)
            p_i.font.color.rgb = C_TEXT_DARK
            p_i.space_before = Pt(3)

    # ==========================================
    # SLIDE 11: Security, HIPAA Privacy & Ethics
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Security, HIPAA Privacy & Clinical Ethics")

    add_card(s11, Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.2), bg_color=C_WHITE)
    tb11 = s11.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "Guiding Ethical & Architectural Safeguards"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    ethics_items = [
        ("Zero-PII Storage Architecture", "Patient names, Medical Record Numbers (MRNs), dates of birth, and identifiable demographic tokens are never transmitted to Supabase or third-party servers. Only aggregated cohort metrics and hyperparameter configurations are persisted."),
        ("Row Level Security (RLS) & Credential Protection", "Privileged database operations are executed strictly via server-side APIs. Database connection strings, service role keys, and backend passwords are never exposed in client JavaScript."),
        ("Medical AI Decision Support Disclaimer", "RARETRACE is an investigative clinical decision support tool designed to flag candidate anomalies for medical review. It does not provide definitive medical diagnoses or prescribe pharmaceutical regimens."),
        ("Algorithmic Auditability", "Every analysis generates an immutable audit record in Supabase detailing dataset name, record count, chosen eps radius, detected noise count, and execution timestamp."),
        ("Mitigation of Screening Bias", "Cohort-relative standardization prevents algorithmic discrimination caused by differing laboratory calibration reference ranges.")
    ]

    for t, d in ethics_items:
        p_t = tf11.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(8)
        p_d = tf11.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_TEXT_MUTED

    # ==========================================
    # SLIDE 12: Conclusion & Future Scope
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Summary, Future Scope & Acknowledgements")

    # Left: Project Achievements
    add_card(s12, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), bg_color=C_WHITE)
    tb12_1 = s12.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf12_1 = tb12_1.text_frame
    tf12_1.word_wrap = True

    p = tf12_1.paragraphs[0]
    p.text = "Key Project Achievements"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    achievements = [
        ("Label-Free Anomaly Mining", "Successfully addresses rare disease diagnostic delays using unsupervised DBSCAN density clustering."),
        ("Transparent Clinical XAI", "Provides clinicians with actionable Z-score feature attributions and 2D PCA spatial maps."),
        ("Automated CDS Suggestions", "Bridges the gap between raw data science outliers and actionable specialist referral pathways."),
        ("Production Cloud Architecture", "Fully containerized, Vercel-ready serverless deployment backed by Supabase PostgreSQL audit storage."),
        ("100% Test Validation", "Comprehensive test suite verifying algorithm correctness, data isolation, and API security.")
    ]
    for t, d in achievements:
        p_t = tf12_1.add_paragraph()
        p_t.text = f"✓ {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = C_TEAL
        p_t.space_before = Pt(6)
        p_d = tf12_1.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Right: Future Work & Credits
    add_card(s12, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2), bg_color=C_BG_LIGHT)
    tb12_2 = s12.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf12_2 = tb12_2.text_frame
    tf12_2.word_wrap = True

    p = tf12_2.paragraphs[0]
    p.text = "Future Scope & Project Credits"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK

    future_pts = [
        ("Multi-Modal Integration", "Incorporate genomic Variant Call Format (VCF) files and clinical unstructured physician notes via Medical LLMs."),
        ("Federated Learning", "Enable collaborative multi-hospital rare disease outlier detection without sharing raw patient data."),
        ("Longitudinal Patient Trajectories", "Apply temporal clustering to track subtle multi-year biomarker deterioration trends.")
    ]
    for t, d in future_pts:
        p_t = tf12_2.add_paragraph()
        p_t.text = f"• {t}: "
        p_t.font.bold = True
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = C_BLUE_ACCENT
        p_t.space_before = Pt(6)
        p_d = tf12_2.add_paragraph()
        p_d.text = f"   {d}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_TEXT_MUTED

    # Credits box
    p_c = tf12_2.add_paragraph()
    p_c.text = "\nThank You! Questions & Discussion"
    p_c.font.size = Pt(14)
    p_c.font.bold = True
    p_c.font.color.rgb = C_NAVY_DARK
    p_c.space_before = Pt(10)

    p_c2 = tf12_2.add_paragraph()
    p_c2.text = "Adithya Vishnubhatla & Pavan Kumar\nFaculty Guide: I V Sai Lakshmi Haritha\nDepartment of CSE • B.Tech Capstone 2026–2027"
    p_c2.font.size = Pt(10.5)
    p_c2.font.color.rgb = C_TEXT_MUTED

    # Save
    prs.save(output_path)
    print(f"[SUCCESS] Presentation generated: {output_path}")

if __name__ == "__main__":
    create_presentation()
