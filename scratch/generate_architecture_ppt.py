"""Script to generate editable PPT slides for VOG-MCI Detection Model Architecture."""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    C_BG = RGBColor(248, 249, 250)         # #F8F9FA Off-white
    C_HEADER = RGBColor(30, 41, 59)        # #1E293B Slate 800
    C_SUBTITLE = RGBColor(100, 116, 139)   # #64748B Slate 500
    
    # Block category colors
    # 1. Preprocessing: Blue
    C_PRE_BG = RGBColor(239, 246, 255)     # Blue 50
    C_PRE_BORDER = RGBColor(147, 197, 253) # Blue 300
    C_PRE_NODE = RGBColor(219, 234, 254)   # Blue 100
    C_PRE_TEXT = RGBColor(30, 58, 138)     # Blue 900
    
    # 2. Backbone / Adapter: Indigo
    C_NET_BG = RGBColor(245, 243, 255)     # Purple 50
    C_NET_BORDER = RGBColor(196, 181, 253) # Purple 300
    C_NET_NODE = RGBColor(237, 233, 254)   # Purple 100
    C_NET_TEXT = RGBColor(76, 29, 149)     # Purple 900
    
    # 3. Task Conditioning & Head: Emerald/Teal
    C_HEAD_BG = RGBColor(240, 253, 250)    # Teal 50
    C_HEAD_BORDER = RGBColor(153, 246, 228)# Teal 300
    C_HEAD_NODE = RGBColor(204, 251, 241)  # Teal 100
    C_HEAD_TEXT = RGBColor(19, 78, 74)     # Teal 900
    
    # 4. Aggregation / Decision: Amber
    C_AGG_BG = RGBColor(254, 252, 232)     # Amber 50
    C_AGG_BORDER = RGBColor(253, 224, 71)  # Amber 300
    C_AGG_NODE = RGBColor(254, 240, 138)   # Amber 200
    C_AGG_TEXT = RGBColor(113, 63, 18)     # Amber 900

    C_ARROW = RGBColor(148, 163, 184)      # Slate 400

    def add_card(slide, left, top, width, height, title, subtitle="", bg_color=None, border_color=None, text_color=None, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
        shape = slide.shapes.add_shape(shape_type, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color or RGBColor(255, 255, 255)
        shape.line.color.rgb = border_color or RGBColor(203, 213, 225)
        shape.line.width = Pt(1.5)
        
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.08)
        tf.margin_top = Inches(0.08)
        tf.margin_bottom = Inches(0.08)
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.name = "Segoe UI"
        p.font.color.rgb = text_color or RGBColor(30, 41, 59)
        p.alignment = PP_ALIGN.CENTER
        
        if subtitle:
            p2 = tf.add_paragraph()
            p2.text = subtitle
            p2.font.size = Pt(9)
            p2.font.name = "Segoe UI"
            p2.font.color.rgb = text_color or RGBColor(71, 85, 105)
            p2.alignment = PP_ALIGN.CENTER
        return shape

    def add_arrow(slide, left, top, width, height, direction="right"):
        shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW if direction=="right" else MSO_SHAPE.DOWN_ARROW, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = C_ARROW
        shape.line.color.rgb = C_ARROW
        return shape

    # ==========================================
    # SLIDE 1: End-to-End Pipeline Overview
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    
    # Title Box
    title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8))
    tf1 = title_box.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "VOG-MCI Detection: End-to-End Architecture (four_error)"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = C_HEADER
    p.font.name = "Segoe UI"
    
    p_sub = tf1.add_paragraph()
    p_sub.text = "From Raw Eye-Tracking Signals to Subject-Level Classification (Kinematics-Free Model)"
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = C_SUBTITLE
    p_sub.font.name = "Segoe UI"

    # Column 1: Preprocessing & CWT
    c1_left = Inches(0.8)
    c1_top = Inches(1.4)
    c1_w = Inches(2.7)
    c1_h = Inches(5.6)
    
    group1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c1_left, c1_top, c1_w, c1_h)
    group1.fill.solid()
    group1.fill.fore_color.rgb = C_PRE_BG
    group1.line.color.rgb = C_PRE_BORDER
    group1.line.width = Pt(1)
    
    lbl1 = slide1.shapes.add_textbox(c1_left + Inches(0.1), c1_top + Inches(0.1), c1_w - Inches(0.2), Inches(0.4))
    lbl1.text_frame.paragraphs[0].text = "1. Preprocessing & CWT"
    lbl1.text_frame.paragraphs[0].font.bold = True
    lbl1.text_frame.paragraphs[0].font.size = Pt(12)
    lbl1.text_frame.paragraphs[0].font.color.rgb = C_PRE_TEXT

    card_w = Inches(2.3)
    card_h = Inches(0.7)
    start_x = c1_left + Inches(0.2)
    y_ptr = c1_top + Inches(0.6)
    
    add_card(slide1, start_x, y_ptr, card_w, card_h, "Raw VOG Recording", "fs ≈ 120Hz (LH, RH, LV, RV, TH, TV)", C_PRE_NODE, C_PRE_BORDER, C_PRE_TEXT)
    y_ptr += card_h
    add_arrow(slide1, start_x + Inches(1.0), y_ptr + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr += Inches(0.35)

    add_card(slide1, start_x, y_ptr, card_w, card_h, "Error & Baseline Correction", "e = Eye - Target (Anti inverted)\nSubtract pre-stim mean (0.2s)", C_PRE_NODE, C_PRE_BORDER, C_PRE_TEXT)
    y_ptr += card_h
    add_arrow(slide1, start_x + Inches(1.0), y_ptr + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr += Inches(0.35)

    add_card(slide1, start_x, y_ptr, card_w, card_h, "Complex Morlet CWT", "15-60 Hz (32 log-bins), cmor4-1\nNearest-zoom to 32 time-bins", C_PRE_NODE, C_PRE_BORDER, C_PRE_TEXT)
    y_ptr += card_h
    add_arrow(slide1, start_x + Inches(1.0), y_ptr + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr += Inches(0.35)

    add_card(slide1, start_x, y_ptr, card_w, card_h, "Sparsification & Z-Score", "Top 15% threshold + 10*log10\nPer-channel Z-score norm", C_PRE_NODE, C_PRE_BORDER, C_PRE_TEXT)
    y_ptr += card_h
    add_arrow(slide1, start_x + Inches(1.0), y_ptr + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr += Inches(0.35)

    add_card(slide1, start_x, y_ptr, card_w, card_h, "Scalogram Tensor X", "Shape: [B, 4, 32, 32]\nChannels: {LH, RH, LV, RV}", RGBColor(255, 255, 255), C_PRE_BORDER, C_PRE_TEXT)

    # Inter-column arrow 1 -> 2
    add_arrow(slide1, c1_left + c1_w + Inches(0.08), Inches(3.9), Inches(0.3), Inches(0.25), "right")

    # Column 2: Backbone Feature Extraction
    c2_left = Inches(3.95)
    c2_top = Inches(1.4)
    c2_w = Inches(3.3)
    c2_h = Inches(5.6)
    
    group2 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c2_left, c2_top, c2_w, c2_h)
    group2.fill.solid()
    group2.fill.fore_color.rgb = C_NET_BG
    group2.line.color.rgb = C_NET_BORDER
    group2.line.width = Pt(1)
    
    lbl2 = slide1.shapes.add_textbox(c2_left + Inches(0.1), c2_top + Inches(0.1), c2_w - Inches(0.2), Inches(0.4))
    lbl2.text_frame.paragraphs[0].text = "2. Vision Backbone (Frozen)"
    lbl2.text_frame.paragraphs[0].font.bold = True
    lbl2.text_frame.paragraphs[0].font.size = Pt(12)
    lbl2.text_frame.paragraphs[0].font.color.rgb = C_NET_TEXT

    start_x2 = c2_left + Inches(0.25)
    card_w2 = Inches(2.8)
    y_ptr2 = c2_top + Inches(0.6)

    add_card(slide1, start_x2, y_ptr2, card_w2, card_h, "ConvAdapter (Trainable)", "Conv2d(4->3, k=(5,1), p=(2,0))\n+ BatchNorm2d + ReLU -> [B, 3, 32, 32]", C_NET_NODE, C_NET_BORDER, C_NET_TEXT)
    y_ptr2 += card_h
    add_arrow(slide1, start_x2 + Inches(1.25), y_ptr2 + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr2 += Inches(0.35)

    add_card(slide1, start_x2, y_ptr2, card_w2, card_h, "Nearest Upsampling (8x)", "F.interpolate(size=(256, 256))\nPreserves edge gradients -> [B, 3, 256, 256]", C_NET_NODE, C_NET_BORDER, C_NET_TEXT)
    y_ptr2 += card_h
    add_arrow(slide1, start_x2 + Inches(1.25), y_ptr2 + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr2 += Inches(0.35)

    add_card(slide1, start_x2, y_ptr2, card_w2, Inches(1.0), "MobileViT-Small (Frozen)", "Pretrained apple/mobilevit-small\nStages 1-5 (MV2 + Local-Global Transformer)\nFinal Feature Map: [B, 640, 8, 8]", RGBColor(255, 255, 255), C_NET_BORDER, C_NET_TEXT)
    y_ptr2 += Inches(1.0)
    add_arrow(slide1, start_x2 + Inches(1.25), y_ptr2 + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr2 += Inches(0.35)

    add_card(slide1, start_x2, y_ptr2, card_w2, card_h, "Global Average Pooling (GAP)", "mean(dim=[2,3]) over spatial dims\nh_vision: [B, 640]", C_NET_NODE, C_NET_BORDER, C_NET_TEXT)

    # Inter-column arrow 2 -> 3
    add_arrow(slide1, c2_left + c2_w + Inches(0.08), Inches(3.9), Inches(0.3), Inches(0.25), "right")

    # Column 3: Task Fusion & Metric Head
    c3_left = Inches(7.4)
    c3_top = Inches(1.4)
    c3_w = Inches(2.9)
    c3_h = Inches(5.6)

    group3 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c3_left, c3_top, c3_w, c3_h)
    group3.fill.solid()
    group3.fill.fore_color.rgb = C_HEAD_BG
    group3.line.color.rgb = C_HEAD_BORDER
    group3.line.width = Pt(1)

    lbl3 = slide1.shapes.add_textbox(c3_left + Inches(0.1), c3_top + Inches(0.1), c3_w - Inches(0.2), Inches(0.4))
    lbl3.text_frame.paragraphs[0].text = "3. Task Fusion & Head"
    lbl3.text_frame.paragraphs[0].font.bold = True
    lbl3.text_frame.paragraphs[0].font.size = Pt(12)
    lbl3.text_frame.paragraphs[0].font.color.rgb = C_HEAD_TEXT

    start_x3 = c3_left + Inches(0.2)
    card_w3 = Inches(2.5)
    y_ptr3 = c3_top + Inches(0.6)

    add_card(slide1, start_x3, y_ptr3, card_w3, card_h, "Task ID Conditioning", "task_id in {0..7}^B\nTaskEmbedding(8, 32) -> e_task: [B, 32]", C_HEAD_NODE, C_HEAD_BORDER, C_HEAD_TEXT)
    y_ptr3 += card_h
    add_arrow(slide1, start_x3 + Inches(1.1), y_ptr3 + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr3 += Inches(0.35)

    add_card(slide1, start_x3, y_ptr3, card_w3, card_h, "Late Feature Fusion", "Concat(h_vision, e_task)\nz: [B, 672]", C_HEAD_NODE, C_HEAD_BORDER, C_HEAD_TEXT)
    y_ptr3 += card_h
    add_arrow(slide1, start_x3 + Inches(1.1), y_ptr3 + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr3 += Inches(0.35)

    add_card(slide1, start_x3, y_ptr3, card_w3, card_h, "Dropout Layer", "nn.Dropout(p=0.3)\nInverted dropout regularization", C_HEAD_NODE, C_HEAD_BORDER, C_HEAD_TEXT)
    y_ptr3 += card_h
    add_arrow(slide1, start_x3 + Inches(1.1), y_ptr3 + Inches(0.05), Inches(0.3), Inches(0.25), "down")
    y_ptr3 += Inches(0.35)

    add_card(slide1, start_x3, y_ptr3, card_w3, Inches(0.85), "CosineLinear Classifier", "Normalized weight & feature: cos(theta)\nLogits: y = 10 * cos(theta) -> [B, 2]", RGBColor(255, 255, 255), C_HEAD_BORDER, C_HEAD_TEXT)

    # Inter-column arrow 3 -> 4
    add_arrow(slide1, c3_left + c3_w + Inches(0.08), Inches(3.9), Inches(0.3), Inches(0.25), "right")

    # Column 4: Subject Decision
    c4_left = Inches(10.45)
    c4_top = Inches(1.4)
    c4_w = Inches(2.2)
    c4_h = Inches(5.6)

    group4 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c4_left, c4_top, c4_w, c4_h)
    group4.fill.solid()
    group4.fill.fore_color.rgb = C_AGG_BG
    group4.line.color.rgb = C_AGG_BORDER
    group4.line.width = Pt(1)

    lbl4 = slide1.shapes.add_textbox(c4_left + Inches(0.1), c4_top + Inches(0.1), c4_w - Inches(0.2), Inches(0.4))
    lbl4.text_frame.paragraphs[0].text = "4. Subject Vote"
    lbl4.text_frame.paragraphs[0].font.bold = True
    lbl4.text_frame.paragraphs[0].font.size = Pt(12)
    lbl4.text_frame.paragraphs[0].font.color.rgb = C_AGG_TEXT

    start_x4 = c4_left + Inches(0.15)
    card_w4 = Inches(1.9)
    y_ptr4 = c4_top + Inches(0.8)

    add_card(slide1, start_x4, y_ptr4, card_w4, Inches(0.8), "Window Softmax", "p_i = Softmax(y)_MCI\nPer-window probability", C_AGG_NODE, C_AGG_BORDER, C_AGG_TEXT)
    y_ptr4 += Inches(0.8)
    add_arrow(slide1, start_x4 + Inches(0.8), y_ptr4 + Inches(0.1), Inches(0.3), Inches(0.25), "down")
    y_ptr4 += Inches(0.45)

    add_card(slide1, start_x4, y_ptr4, card_w4, Inches(1.1), "Weighted Soft Vote", "P_s = sum(w_t * p_i) / sum(w_t)\nWeights: [0, 0, 0, 1.5,\n0, 1.5, 3.0, 2.5]", C_AGG_NODE, C_AGG_BORDER, C_AGG_TEXT)
    y_ptr4 += Inches(1.1)
    add_arrow(slide1, start_x4 + Inches(0.8), y_ptr4 + Inches(0.1), Inches(0.3), Inches(0.25), "down")
    y_ptr4 += Inches(0.45)

    add_card(slide1, start_x4, y_ptr4, card_w4, Inches(0.9), "Final Classification", "P_s >= 0.5 -> MCI (Class 1)\nP_s < 0.5 -> HC (Class 0)", RGBColor(255, 255, 255), C_AGG_BORDER, C_AGG_TEXT)


    # ==========================================
    # SLIDE 2: Deep Dive: TransferMobileViTClassifier Architecture
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)

    # Title Box
    title_box2 = slide2.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8))
    tf2 = title_box2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "TransferMobileViTClassifier: Mathematical Architecture"
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = C_HEADER
    p2.font.name = "Segoe UI"

    p2_sub = tf2.add_paragraph()
    p2_sub.text = "Mathematical Transformations, Tensor Shapes, and Sub-Block Operations"
    p2_sub.font.size = Pt(12)
    p2_sub.font.color.rgb = C_SUBTITLE
    p2_sub.font.name = "Segoe UI"

    # Block 1: ConvAdapter & Upsample (Left)
    b1_x = Inches(0.8)
    b1_y = Inches(1.4)
    b1_w = Inches(3.6)
    b1_h = Inches(5.6)
    
    g_b1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b1_x, b1_y, b1_w, b1_h)
    g_b1.fill.solid()
    g_b1.fill.fore_color.rgb = C_PRE_BG
    g_b1.line.color.rgb = C_PRE_BORDER
    
    t_b1 = slide2.shapes.add_textbox(b1_x + Inches(0.15), b1_y + Inches(0.1), b1_w - Inches(0.3), Inches(0.4))
    t_b1.text_frame.paragraphs[0].text = "1. Input Adapter & Upsampling"
    t_b1.text_frame.paragraphs[0].font.bold = True
    t_b1.text_frame.paragraphs[0].font.size = Pt(12)
    t_b1.text_frame.paragraphs[0].font.color.rgb = C_PRE_TEXT

    by = b1_y + Inches(0.6)
    bw = Inches(3.2)
    bx = b1_x + Inches(0.2)

    add_card(slide2, bx, by, bw, Inches(0.8), "Input Scalogram X", "X in R^{B x 4 x 32 x 32}\nChannels: [LH, RH, LV, RV]", RGBColor(255, 255, 255), C_PRE_BORDER, C_PRE_TEXT)
    by += Inches(0.8)
    add_arrow(slide2, bx + Inches(1.45), by + Inches(0.05), Inches(0.3), Inches(0.2), "down")
    by += Inches(0.3)

    add_card(slide2, bx, by, bw, Inches(1.1), "Asymmetric Conv2d", "W in R^{3 x 4 x 5 x 1}, padding=(2, 0)\nVertical Sobel-like edge filtering\nOutput U in R^{B x 3 x 32 x 32}", C_PRE_NODE, C_PRE_BORDER, C_PRE_TEXT)
    by += Inches(1.1)
    add_arrow(slide2, bx + Inches(1.45), by + Inches(0.05), Inches(0.3), Inches(0.2), "down")
    by += Inches(0.3)

    add_card(slide2, bx, by, bw, Inches(0.9), "BatchNorm2d + ReLU", "Normalize across channels + max(0, .)\nX_adapt in R^{B x 3 x 32 x 32}", C_PRE_NODE, C_PRE_BORDER, C_PRE_TEXT)
    by += Inches(0.9)
    add_arrow(slide2, bx + Inches(1.45), by + Inches(0.05), Inches(0.3), Inches(0.2), "down")
    by += Inches(0.3)

    add_card(slide2, bx, by, bw, Inches(1.1), "Nearest Interpolation 8x", "F.interpolate(X_adapt, size=(256, 256))\nX_up(b,c,y,x) = X_adapt(b,c,floor(y/8),floor(x/8))\nOutput: [B, 3, 256, 256]", RGBColor(255, 255, 255), C_PRE_BORDER, C_PRE_TEXT)

    # Arrow 1 -> 2
    add_arrow(slide2, b1_x + b1_w + Inches(0.08), Inches(3.9), Inches(0.3), Inches(0.25), "right")

    # Block 2: MobileViT-Small Backbone Internal Blocks (Middle)
    b2_x = Inches(4.85)
    b2_y = Inches(1.4)
    b2_w = Inches(4.3)
    b2_h = Inches(5.6)

    g_b2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b2_x, b2_y, b2_w, b2_h)
    g_b2.fill.solid()
    g_b2.fill.fore_color.rgb = C_NET_BG
    g_b2.line.color.rgb = C_NET_BORDER

    t_b2 = slide2.shapes.add_textbox(b2_x + Inches(0.15), b2_y + Inches(0.1), b2_w - Inches(0.3), Inches(0.4))
    t_b2.text_frame.paragraphs[0].text = "2. Frozen MobileViT-Small Backbone Details"
    t_b2.text_frame.paragraphs[0].font.bold = True
    t_b2.text_frame.paragraphs[0].font.size = Pt(12)
    t_b2.text_frame.paragraphs[0].font.color.rgb = C_NET_TEXT

    b2y = b2_y + Inches(0.6)
    b2w = Inches(3.9)
    b2x = b2_x + Inches(0.2)

    add_card(slide2, b2x, b2y, b2w, Inches(0.9), "Inverted Residual (MV2) Block", "F_1 = Swish(BN(Conv1x1(F)))\nF_2 = Swish(BN(DWConv3x3(F_1)))\nF_out = F + BN(Conv1x1(F_2))  (Residual if stride=1)", C_NET_NODE, C_NET_BORDER, C_NET_TEXT)
    b2y += Inches(0.9)
    add_arrow(slide2, b2x + Inches(1.8), b2y + Inches(0.04), Inches(0.3), Inches(0.18), "down")
    b2y += Inches(0.25)

    add_card(slide2, b2x, b2y, b2w, Inches(1.5), "MobileViT Transformer Block", "Local Rep: F_L = Conv1x1(Swish(BN(Conv3x3(F_in))))\nUnfold to PxN tokens: F_U = Unfold(F_L)\nGlobal Self-Attention: F_G = Transformer(F_U)\nFold + Pointwise: F_proj = Swish(BN(Conv1x1(Fold(F_G))))\nFusion: F_out = Conv3x3(Concat(F_in, F_proj))", RGBColor(255, 255, 255), C_NET_BORDER, C_NET_TEXT)
    b2y += Inches(1.5)
    add_arrow(slide2, b2x + Inches(1.8), b2y + Inches(0.04), Inches(0.3), Inches(0.18), "down")
    b2y += Inches(0.25)

    add_card(slide2, b2x, b2y, b2w, Inches(0.85), "Hierarchical Stages (Stem -> S5)", "Stem (s=2) -> S1 (32d) -> S2 (48d) ->\nS3 (64d, L=2) -> S4 (80d, L=4) -> S5 (96d, L=3)\nHead Conv1x1 -> [B, 640, 8, 8]", C_NET_NODE, C_NET_BORDER, C_NET_TEXT)
    b2y += Inches(0.85)
    add_arrow(slide2, b2x + Inches(1.8), b2y + Inches(0.04), Inches(0.3), Inches(0.18), "down")
    b2y += Inches(0.25)

    add_card(slide2, b2x, b2y, b2w, Inches(0.7), "Global Average Pooling (GAP)", "h_vision = (1/64) * sum_{h,w} F_last(b, c, h, w)\nh_vision in R^{B x 640}", C_NET_NODE, C_NET_BORDER, C_NET_TEXT)

    # Arrow 2 -> 3
    add_arrow(slide2, b2_x + b2_w + Inches(0.08), Inches(3.9), Inches(0.3), Inches(0.25), "right")

    # Block 3: Head, Metric Learning & Loss (Right)
    b3_x = Inches(9.5)
    b3_y = Inches(1.4)
    b3_w = Inches(3.0)
    b3_h = Inches(5.6)

    g_b3 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b3_x, b3_y, b3_w, b3_h)
    g_b3.fill.solid()
    g_b3.fill.fore_color.rgb = C_HEAD_BG
    g_b3.line.color.rgb = C_HEAD_BORDER

    t_b3 = slide2.shapes.add_textbox(b3_x + Inches(0.15), b3_y + Inches(0.1), b3_w - Inches(0.3), Inches(0.4))
    t_b3.text_frame.paragraphs[0].text = "3. Metric Head & Decision"
    t_b3.text_frame.paragraphs[0].font.bold = True
    t_b3.text_frame.paragraphs[0].font.size = Pt(12)
    t_b3.text_frame.paragraphs[0].font.color.rgb = C_HEAD_TEXT

    b3y = b3_y + Inches(0.6)
    b3w = Inches(2.6)
    b3x = b3_x + Inches(0.2)

    add_card(slide2, b3x, b3y, b3w, Inches(0.9), "Task Conditioning", "E in R^{8 x 32} (Lookup table)\ne_task = E[task_id] in R^{B x 32}\nz = [h_vision || e_task] in R^{B x 672}", C_HEAD_NODE, C_HEAD_BORDER, C_HEAD_TEXT)
    b3y += Inches(0.9)
    add_arrow(slide2, b3x + Inches(1.15), b3y + Inches(0.05), Inches(0.3), Inches(0.2), "down")
    b3y += Inches(0.3)

    add_card(slide2, b3x, b3y, b3w, Inches(0.75), "Inverted Dropout", "z_tilde = z * m / (1 - 0.3)\nRegularization (p = 0.3)", C_HEAD_NODE, C_HEAD_BORDER, C_HEAD_TEXT)
    b3y += Inches(0.75)
    add_arrow(slide2, b3x + Inches(1.15), b3y + Inches(0.05), Inches(0.3), Inches(0.2), "down")
    b3y += Inches(0.3)

    add_card(slide2, b3x, b3y, b3w, Inches(1.2), "CosineLinear Classifier", "W in R^{2 x 672} (Class prototypes)\nz_hat = z / ||z||_2,  w_hat_c = w_c / ||w_c||_2\ny_c = 10.0 * (z_hat . w_hat_c)\ny_c = 10.0 * cos(theta_c)", RGBColor(255, 255, 255), C_HEAD_BORDER, C_HEAD_TEXT)
    b3y += Inches(1.2)
    add_arrow(slide2, b3x + Inches(1.15), b3y + Inches(0.05), Inches(0.3), Inches(0.2), "down")
    b3y += Inches(0.3)

    add_card(slide2, b3x, b3y, b3w, Inches(0.85), "Softmax Probability", "p_i = 1 / (1 + exp(-(y_1 - y_0)))\nProbability of MCI for window i", C_HEAD_NODE, C_HEAD_BORDER, C_HEAD_TEXT)

    # Save
    prs.save(str(output_path))
    print(f"Saved editable presentation to {output_path}")

if __name__ == "__main__":
    out_file = Path("model_architecture_diagram.pptx").resolve()
    create_deck(out_file)
