# --- START OF FILE darts_app.py ---

import streamlit as st
import numpy as np
import cv2
import os
import time
import shutil
import google.generativeai as genai
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import io 
import tempfile

# =========================================================
# 1. ページの基本設定 (***必須：stコマンドの最初に置く***)
# =========================================================
st.set_page_config(
    page_title="DARTS Re:CODE", 
    page_icon="🎯",
    layout="wide"
)

# =========================================================
# 2. セッションステートの初期化
# =========================================================
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'dominant_arm_select' not in st.session_state:
    st.session_state.dominant_arm_select = "右利き"
if 'current_barrel_type' not in st.session_state:
    st.session_state.current_barrel_type = "選択してください"
if 'current_barrel_weight' not in st.session_state:
    st.session_state.current_barrel_weight = ""
if 'current_shaft_length' not in st.session_state:
    st.session_state.current_shaft_length = "選択してください"
if 'current_flight_shape' not in st.session_state:
    st.session_state.current_flight_shape = "選択してください"
if 'q0_2' not in st.session_state:
    st.session_state.q0_2 = "A: とにかく最初は安く始めたい（〜5,000円くらい）"
if 'q1' not in st.session_state:
    st.session_state.q1 = "A: 指の全体でしっかり握り、手のひらに近い方で重さを感じる"
if 'q2' not in st.session_state:
    st.session_state.q2 = "A: 狙った場所に向かって、腕をしっかり「押し出す」ように伸ばすのが自然"
if 'q3' not in st.session_state:
    st.session_state.q3 = "A: 狙った場所より少し下に刺さることが多い"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'uploaded_photo' not in st.session_state:
    st.session_state.uploaded_photo = None

# =========================================================
# 3. カスタムCSSの定義とGoogle Fontsのインポート
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mochiy+Pop+P+One&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
.stApp { background-image: url("https://github.com/k024c1009-ui/darts_app/raw/main/Gemini_Generated_Image_6uf6ec6uf6ec6uf6.png"); background-size: cover; background-attachment: fixed; background-position: center; background-color: #191919; color: #E0E0E0; }
h1 { font-family: 'Mochiy Pop P One', 'Orbitron', sans-serif !important; font-weight: 700; color: #00FFFF; text-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 15px rgba(0, 255, 255, 0.5); }
h2, h3 { font-family: 'Mochiy Pop P One', 'Orbitron', sans-serif !important; font-weight: 700; color: #00FFFF; text-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 15px rgba(0, 255, 255, 0.5): }
div[data-testid="stAppViewBlock"] > section:nth-child(2) > div:first-child, div.block-container { background: rgba(0, 0, 0, 0.75); padding: 20px; border-radius: 10px; }
div[data-testid="stVerticalBlock"] > div:first-child > div:first-child { background-color: black; padding: 20px; margin-bottom: 20px; border-radius: 0 0 10px 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); }
html, body, .stApp, .stRadio label, .stSelectbox label, p, li { font-family: 'Montserrat', sans-serif; font-weight: 400; }
p, li, .stText { color: #E0E0E0 !important; text-shadow: 0 0 2px #E0E0E0, 0 0 5px rgba(255, 255, 255, 0.2); font-size: 1.05em; }
.stRadio > label, .stSelectbox > label { color: #FFFFAA !important; text-shadow: 0 0 5px #FFD700; font-size: 1.2em; font-weight: 600; }
.stButton>button { background-color: #00FFFF; color: #191919; border: 2px solid #00FFFF; border-radius: 8px; font-weight: bold; padding: 10px 20px; transition: all 0.2s; }
.stButton>button:hover { background-color: #191919; color: #00FFFF; box-shadow: 0 0 10px #00FFFF; }
div[data-testid="stSelectbox"] div[role="listbox"] { background-color: white !important; }
.stSelectbox div[role="listbox"] span, .stSelectbox div[role="listbox"] p { color: black !important; }
div[data-testid="stAlert"] div[role="alert"].stAlert.info { background-color: rgba(0, 255, 255, 0.1); border-left: 5px solid #00FFFF; }
div[data-testid="stAlert"] div[role="alert"].stAlert.warning { background-color: rgba(255, 165, 0, 0.1); border-left: 5px solid #FFD700; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. ヘルパー関数 (タイポ修正済み)
# =========================================================
def calculate_angle(a: list, b: list, c: list) -> float:
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def process_video_for_analysis(video_path: str, dominant_arm: str, output_dir: str) -> (list, str):
    elbow_angles = []; over_path = os.path.join(output_dir, "overlay_output.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return [], None
    fps = cap.get(cv2.CAP_PROP_FPS); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(over_path, fourcc, fps, (width, height))
    mp_pose = mp.solutions.pose; mp_drawing = mp.solutions.drawing_utils
    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0; pbar = st.progress(0, text="フォーム動画を解析中...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image.flags.writeable = False; results = pose.process(image); image.flags.writeable = True; image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    try:
                        # ★★★ タイポ修正: results.pose_ landmarks -> results.pose_landmarks ★★★
                        landmarks = results.pose_landmarks.landmark
                        shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_SHOULDER; elbow_landmark = mp_pose.PoseLandmark.RIGHT_ELBOW if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_ELBOW; wrist_landmark = mp_pose.PoseLandmark.RIGHT_WRIST if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_WRIST
                        shoulder = [landmarks[shoulder_landmark.value].x, landmarks[shoulder_landmark.value].y]; elbow = [landmarks[elbow_landmark.value].x, landmarks[elbow_landmark.value].y]; wrist = [landmarks[wrist_landmark.value].x, landmarks[wrist_landmark.value].y]
                        angle_deg = calculate_angle(shoulder, elbow, wrist); elbow_angles.append(angle_deg)
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    except Exception: pass
                out.write(image); frame_count += 1; progress_value = min(frame_count / (total_frames + 1), 1.0); pbar.progress(progress_value, text=f"フォーム動画を解析中... {int(progress_value * 100)}%")
            pbar.empty()
    finally:
        cap.release(); out.release()
    return elbow_angles, over_path

# =========================================================
# 5. UI（ページ遷移ロジック）
# =========================================================

# --- ページ1: ウェルカム ＆ マイダーツ情報 ---
if st.session_state.page == 1:
    st.title("🎯 DARTS Re:CODE")
    st.subheader("あなたのダーツ、次なる進化へ！【マイダーツ深掘り診断】")
    st.write("今のマイダーツに、もう少しフィット感が欲しいと思いませんか？")
    st.write("この診断では、**あなたが普段使っているマイダーツの情報**と**投げ方の感覚**を深掘りし、**フォームの特性**、そして**予算**も考慮して、あなたに最適な『次の一本』や『セッティングのヒント』を提案します！")
    st.markdown("---")
    st.header("ステップ1: あなたのマイダーツ情報を教えてください")
    st.write("あなたが普段使っているダーツについて教えてください。よりパーソナルな診断に役立てます。")
    st.radio("Q1-1: ダーツを投げる利き腕はどちらですか？", ("右利き", "左利き"), key="dominant_arm_select")
    st.selectbox("Q1-2: 今お使いのバレルの形状は？", ["選択してください", "ストレート", "トルピード", "砲弾", "その他・わからない"], key="current_barrel_type")
    st.text_input("Q1-3: 今お使いのバレルの重さ（g）は？ (例: 18.0)", key="current_barrel_weight")
    st.write("※ご自身のバレルの重さが不明な場合は、およそ18g〜20gで試投される方が多いです。")
    st.selectbox("Q1-4: 今お使いのシャフトの長さは？", ["選択してください", "ショート", "ミディアム", "ロング", "その他・わからない"], key="current_shaft_length")
    st.selectbox("Q1-5: 今お使いのフライトの形状は？", ["選択してください", "スタンダード", "シェイプ", "カイト", "スリム", "その他・わからない"], key="current_flight_shape")
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("次へ進む →", use_container_width=True):
            st.session_state.page = 2
            st.rerun()

# --- ページ2: 予算とダーツ感覚のヒアリング ---
elif st.session_state.page == 2:
    st.header("ステップ2: あなたのダーツについて教えてください")
    st.radio("Q2-1: マイダーツにかけられる予算のイメージはありますか？", ("A: とにかく最初は安く始めたい（〜5,000円くらい）", "B: 初心者だけど、長く使えるものがほしい（5,000円〜15,000円くらい）", "C: デザインも性能も妥協したくない（15,000円円以上）", "D: まずは診断結果を見てから決めたい（予算は後で考える）"), key="q0_2")
    st.markdown("---")
    st.radio("Q2-2: ダーツを握るとき、指のどこにダーツの重さを感じますか？", ("A: 指の全体でしっかり握り、手のひらに近い方で重さを感じる", "B: 指の先端の方でバランスを取りながら、軽やかに感じる", "C: ダーツの真ん中あたりを指で探して、そこが一番しっくりくる"), key="q1")
    st.radio("Q2-3: ダーツを投げた後、腕はどのように伸びていくのが一番自然だと感じますか？", ("A: 狙った場所に向かって、腕をしっかり「押し出す」ように伸ばすのが自然", "B: 腕を「振り子」のように、力を抜いて自然に「振り抜く」のが自然", "C: 的に指を指すように、最後まで「まっすぐ」腕を伸ばしきるのが自然"), key="q2")
    st.radio("Q2-4: ダーツがボードに刺さったとき、狙った場所からどうズレることが一番多いですか？", ("A: 狙った場所より少し下に刺さることが多い", "B: 狙った場所より少し上に刺さることが多い", "C: 上下というより、左右にバラけることが多い"), key="q3")
    st.markdown("---")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("← 戻る", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col3:
        if st.button("次へ進む →", use_container_width=True):
            st.session_state.page = 3
            st.rerun()

# --- ページ3: 動画・写真のアップロード ---
elif st.session_state.page == 3:
    st.header("ステップ3: フォームと着弾点をアップロード")
    st.info("💡 **【重要】フォーム分析のための撮影ガイドライン**\n\n1. ダーツを投げる腕の**真横**から撮影してください。\n2. **肩の高さ**にカメラを設置するのが理想です。\n3. 体全体がフレームに収まるようにしてください。\n4. **逆光を避け**、明るい場所で撮影してください。")
    st.file_uploader("フォーム動画をアップロード", type=["mp4", "mov", "avi"], key="uploaded_file")
    if st.session_state.uploaded_file:
        st.video(st.session_state.uploaded_file)
    st.markdown("---")
    st.info("💡 **【重要】着弾点分析のための注意点**\n\n1. 写真はダーツボードの**真正面**から、歪みがないように撮影してください。\n2. **全てのダーツ**が刺さった状態のものが理想です。")
    st.file_uploader("着弾点の写真をアップロード", type=["jpg", "jpeg", "png"], key="uploaded_photo")
    if st.session_state.uploaded_photo:
        st.image(st.session_state.uploaded_photo, caption="アップロードされた着弾点写真", use_column_width=True)
    st.markdown("---")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("← 戻る", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with col3:
        if st.button("診断を開始する！", type="primary", use_container_width=True):
            if st.session_state.uploaded_file is None:
                st.error("診断にはフォーム動画のアップロードが必須です。")
            else:
                st.session_state.page = 4
                st.rerun()
                
# --- ページ4: 診断中・結果表示 ---
elif st.session_state.page == 4:
    temp_dir = tempfile.mkdtemp()
    try:
        video_file = st.session_state.uploaded_file
        video_path = os.path.join(temp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        try:
            if "GEMINI_API_KEY" not in st.secrets:
                st.warning("⚠️ Google Gemini APIキーが設定されていません。")
                st.stop()
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        except Exception as e:
            st.error(f"Gemini APIの設定中にエラーが発生しました: {e}")
            st.stop()
        with st.spinner("診断中... AIがあなたのフォームとダーツ感覚、好みを解析しています"):
            prompt_parts = [
                "あなたはダーツの専門家であり、ダーツプレイヤーのパフォーマンス向上をサポートするAIです。", "ユーザーは自分のマイダーツを持っており、より深くダーツを理解し、次のステップに進みたいと考えています。", "以下のユーザーの現在のマイダーツ情報、ダーツを投げた際の感覚、そしてフォームの特性から、", "ユーザーに最適なダーツバレル（形状、重さ、カットの特徴）、または現在のセッティングからの改善点や次の試すべきダーツの方向性を、具体的な理由とともに提案してください。", "回答は、経験者にも納得感があり、かつ親しみやすい言葉遣いでお願いします。", "---ユーザーの現在のマイダーツ情報---", f"バレルの形状: {st.session_state.current_barrel_type}", f"バレルの重さ: {st.session_state.current_barrel_weight}g", f"シャフトの長さ: {st.session_state.current_shaft_length}", f"フライトの形状: {st.session_state.current_flight_shape}", "---ユーザーのダーツ感覚---", f"Q1（握る感覚）: {st.session_state.q1}", f"Q2（腕の伸び方）: {st.session_state.q2}", f"Q3（ダーツのズレ）: {st.session_state.q3}",
            ]
            q0_2_val = st.session_state.q0_2
            budget_info = ""
            if q0_2_val.startswith("A:"): budget_info = "予算は5,000円くらいで、最初は安く始めたい。"
            elif q0_2_val.startswith("B:"): budget_info = "予算は5,000円〜15,000円で、長く使えるものが欲しい。"
            elif q0_2_val.startswith("C:"): budget_info = "予算は15,000円以上で、デザインも性能も妥協したくない。"
            else: budget_info = "予算は診断結果を見てから決めたい。"
            prompt_parts.append(f"---ユーザーの予算に関する希望---\n{budget_info}")
            model_name = ""
            if st.session_state.uploaded_photo:
                model_name = 'gemini-pro-vision'
                photo_image = Image.open(st.session_state.uploaded_photo)
                prompt_parts.extend([
                    "\n---着弾点写真による分析---", "ユーザーから提供されたダーツボードの写真です。この画像から着弾点のグルーピング（集まり具合）、上下左右の偏り、回転の有無などを専門家の視点で分析し、その結果を診断に含めてください。例えば、「全体的にブルの左下に集まっているので、リリースが早いか、少し引っ掛けている可能性があります」といった具体的な考察をしてください。", photo_image
                ])
            else:
                model_name = 'gemini-pro'
            
            dominant_arm_val = st.session_state.dominant_arm_select
            st.subheader(f"📊 詳細フォーム分析（あなたの{dominant_arm_val}の肘の角度）")
            st.write(f"動画からあなたの{dominant_arm_val}の肘の角度を抽出し、その推移をグラフで表示します。")
            elbow_angles, over_path = process_video_for_analysis(video_path, dominant_arm_val, temp_dir)
            if elbow_angles:
                fig, ax = plt.subplots(); ax.plot(elbow_angles); ax.set_title("肘の角度の時系列変化"); ax.set_xlabel("フレーム"); ax.set_ylabel("角度 (度)"); st.pyplot(fig); plt.close(fig)
                avg_angle = np.mean(elbow_angles); std_dev_angle = np.std(elbow_angles)
                st.write(f"**平均肘角度:** {avg_angle:.2f}度")
                st.write(f"**肘角度の標準偏差（ブレ）:** {std_dev_angle:.2f}度")
                st.info("※肘の角度のブレが小さいほど、安定したリリースができていることを示します。")
                prompt_parts.extend([
                    f"\n---フォーム分析（数値データ）---", f"平均肘角度: {avg_angle:.2f}度", f"肘角度の標準偏差（ブレ）: {std_dev_angle:.2f}度。この数値が小さいほどフォームが安定していると解釈してください。"
                ])
                if over_path and os.path.exists(over_path):
                    st.subheader("👀 あなたのフォーム（骨格オーバーレイ）"); st.video(over_path)
            else:
                st.warning("動画からフォームのランドマークを検出できませんでした。撮影ガイドラインを参考に、もう一度お試しください。")
            
            model = genai.GenerativeModel(model_name)
            generation_config = genai.GenerationConfig(temperature=0.7)
            try:
                response = model.generate_content(prompt_parts, generation_config=generation_config)
                st.subheader("🎁 AIによるあなたの診断結果")
                st.success(response.text)
            except Exception as e:
                st.error(f"AI診断中にエラーが発生しました。エラー詳細: {e}")
            st.markdown("---")
            st.write("### 🎯 診断結果を参考に、次の一歩を踏み出そう！")
            st.write("この診断は、AIがあなたの**深い感覚とフォームの特性**から推測したものです。")
            st.write("最終的には、実際にダーツショップなどで**専門スタッフに相談**し、**様々なダーツを『試投』**して、あなたの手に最も馴染む一本を見つけることが重要です。")
            st.write("この診断結果をヒントに、ぜひあなたのダーツを次のレベルへと進化させてくださいね！")
            if st.button("もう一度診断する"):
                # セッションステートをクリアして最初のページに戻る
                keys_to_delete = [key for key in st.session_state.keys() if key != 'page']
                for key in keys_to_delete:
                    del st.session_state[key]
                st.session_state.page = 1
                st.rerun()
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)