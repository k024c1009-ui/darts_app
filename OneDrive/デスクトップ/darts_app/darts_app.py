import streamlit as st
import numpy as np
import cv2
import os
import time
import shutil
import google.generativeai as genai
import mediapipe as mp
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import io # Matplotlibの図をPIL Imageに変換するために必要
import gc # ファイルロック解除のため

# =========================================================
# 1. ページの基本設定 (***必須：stコマンドの最初に置く***)
# =========================================================
st.set_page_config(
    page_title="【深掘り診断】あなたのダーツ、次なる進化へ！",
    page_icon="🎯",
    layout="wide"
)

# =========================================================
# 2. セッションステートの初期化
# =========================================================

# 着弾点入力機能のキー管理は不要になったため削除
# if 'hit_points' not in st.session_state:
#     st.session_state.hit_points = []
# if 'reset_count' not in st.session_state:
#     st.session_state.reset_count = 0


# =========================================================
# 3. カスタムCSSの定義とGoogle Fontsのインポート
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Mochiy+Pop+P+One&display=swap');

/* 🌟【最重要】アプリ全体のテーマとフォント設定 🌟 */
.stApp {
    /* 背景画像をアプリ全体に適用 */
    background-image: url("https://github.com/k024c1009-ui/darts_app/raw/main/Gemini_Generated_Image_6uf6ec6uf6ec6uf6.png"); 
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    
    /* 文字のトーン */
    background-color: #191919; 
    color: #E0E0E0; 
}

/* 🌟 見出し・タイトルを Orbitron に変更し、ネオン効果を強化 🌟 */
h1 {
    font-family: 'Mochiy Pop P One', 'Orbitron', sans-serif; 
    font-weight: 700;
    color: #00FFFF;
    text-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 15px rgba(0, 255, 255, 0.5);
}

h2, h3 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 600;
    color: #00FFFF;
    text-shadow: 0 0 3px #00FFFF;
}

/* 🌟 メインコンテンツエリアの半透明の黒い背景を適用 🌟 */
/* st.title の下からコンテンツ全体を覆う */
div[data-testid="stAppViewBlock"] > section:nth-child(2) > div:first-child,
div.block-container { 
    background: rgba(0, 0, 0, 0.75); /* 濃い半透明の黒を適用 */
    padding: 20px; 
    border-radius: 10px;
}

/* 🌟 タイトルエリアの可読性確保（完全な黒で塗りつぶし） 🌟 */
div[data-testid="stVerticalBlock"] > div:first-child > div:first-child {
    background-color: black; 
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 0 0 10px 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
}


/* 🌟 標準テキストとラベルの装飾（Montserrat） 🌟 */

/* アプリ全体と要素へのフォント適用 */
html, body, .stApp, .stRadio label, .stSelectbox label, p, li {
    font-family: 'Montserrat', sans-serif; 
    font-weight: 400;
}

/* 全ての標準テキストの影を調整 */
p, li, .stText { 
    color: #E0E0E0 !important; 
    text-shadow: 0 0 2px #E0E0E0, 0 0 5px rgba(255, 255, 255, 0.2); 
    font-size: 1.05em;
}

/* ラジオボタン、セレクトボックスなどのラベル */
.stRadio > label, .stSelectbox > label {
    color: #FFFFAA !important; /* 質問文を黄色系のネオンカラーに */
    text-shadow: 0 0 5px #FFD700; /* ゴールド系の影 */
    font-size: 1.2em;
    font-weight: 600;
}


/* 🌟 UI要素の色調整 🌟 */

/* 診断ボタンのカスタマイズ */
.stButton>button {
    background-color: #00FFFF;
    color: #191919;
    border: 2px solid #00FFFF;
    border-radius: 8px;
    font-weight: bold;
    padding: 10px 20px;
    transition: all 0.2s;
}
.stButton>button:hover {
    background-color: #191919;
    color: #00FFFF;
    box-shadow: 0 0 10px #00FFFF;
}

/* ドロップダウンメニューのリストの背景（選択肢の文字を黒くするため） */
div[data-testid="stSelectbox"] div[role="listbox"] {
    background-color: white !important; 
}
.stSelectbox div[role="listbox"] span,
.stSelectbox div[role="listbox"] p {
    color: black !important; /* 選択肢の文字を黒にする */
}

/* 情報・警告ボックスのテーマ化 */
div[data-testid="stAlert"] div[role="alert"].stAlert.info {
    background-color: rgba(0, 255, 255, 0.1); 
    border-left: 5px solid #00FFFF; 
}
div[data-testid="stAlert"] div[role="alert"].stAlert.warning {
    background-color: rgba(255, 165, 0, 0.1); 
    border-left: 5px solid #FFD700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. フォーム解析のためのヘルパー関数
# =========================================================
def calculate_angle(a, b, c):
    """3つのランドマークから角度を計算する"""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_video_for_analysis(video_path, dominant_arm, mp_pose, mp_drawing):
    """動画を解析し、肘の角度リストとオーバーレイ動画パスを返す"""
    elbow_angles = []
    over_path = os.path.join(os.path.dirname(video_path), "overlay_output.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return elbow_angles, None, cap, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(over_path, fourcc, fps, (width, height))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        pbar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_SHOULDER
                    elbow_landmark = mp_pose.PoseLandmark.RIGHT_ELBOW if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_ELBOW
                    wrist_landmark = mp_pose.PoseLandmark.RIGHT_WRIST if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_WRIST
                    
                    shoulder = [landmarks[shoulder_landmark.value].x, landmarks[shoulder_landmark.value].y]
                    elbow = [landmarks[elbow_landmark.value].x, landmarks[elbow_landmark.value].y]
                    wrist = [landmarks[wrist_landmark.value].x, landmarks[wrist_landmark.value].y]
                        
                    angle_deg = calculate_angle(shoulder, elbow, wrist)
                    elbow_angles.append(angle_deg)
                    
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                                
                except Exception:
                    pass 
            
            out.write(image)
            
            frame_count += 1
            pbar.progress(min(frame_count / (total_frames + 1), 1.0))
        
        pbar.empty()
        
    return elbow_angles, over_path, cap, out

# --- アプリのメイン表示 ---
st.title("🎯 あなたのダーツ、次なる進化へ！【マイダーツ深掘り診断】")
st.write("今のマイダーツに、もう少しフィット感が欲しいと思いませんか？")
st.write("この診断では、**あなたが普段使っているマイダーツの情報**と**投げ方の感覚**を深掘りし、**フォームの特性**、そして**予算**も考慮して、あなたに最適な『次の一本』や『セッティングのヒント』を提案します！")
st.write("さあ、あなたのダーツを次のレベルへと進化させましょう！")

st.markdown("---")


# --- ステップ0: あなたのマイダーツ情報を教えてください ---
st.header("ステップ0: あなたのマイダーツ情報を教えてください")
st.write("あなたが普段使っているダーツについて、差し支えなければ教えてください。よりパーソナルな診断に役立てます。")

dominant_arm = st.radio(
    "Q0-0: ダーツを投げる利き腕はどちらですか？",
    ("右利き", "左利き"),
    key="dominant_arm_select"
    )

# 現在のバレル情報
current_barrel_type = st.selectbox(
    "今お使いのバレルの形状は？",
    ["選択してください", "ストレート", "トルピード", "砲弾", "その他・わからない"],
    key="current_barrel_type"
)
current_barrel_weight = st.text_input(
    "今お使いのバレルの重さ（g）は？ (例: 18.0)",
    key="current_barrel_weight"
)
st.write("※ご自身のバレルの重さが不明な場合は、およそ18g〜20gで試投される方が多いです。")

# 現在のシャフト情報
current_shaft_length = st.selectbox(
    "今お使いのシャフトの長さは？",
    ["選択してください", "ショート", "ミディアム", "ロング", "その他・わからない"],
    key="current_shaft_length"
)

# 現在のフライト情報
current_flight_shape = st.selectbox(
    "今お使いのフライトの形状は？",
    ["選択してください", "スタンダード", "シェイプ", "カイト", "スリム", "その他・わからない"],
    key="current_flight_shape"
)

st.markdown("---")


# --- ステップ1: 予算のイメージを教えてください ---
st.header("ステップ1: 予算のイメージを教えてください")
q0_2 = st.radio(
    "Q0-2: マイダーツにかけられる予算のイメージはありますか？",
    ("A: とにかく最初は安く始めたい（〜5,000円くらい）",
      "B: 初心者だけど、長く使えるものがほしい（5,000円〜15,000円くらい）",
      "C: デザインも性能も妥協したくない（15,000円円以上）",
      "D: まずは診断結果を見てから決めたい（予算は後で考える）")
)

st.markdown("---")


# --- ステップ2: ダーツ感覚診断 ---
st.header("ステップ2: あなたの「ダーツ感覚」を教えてください")

# Q1: ダーツを握る際、指のどこにダーツの重さを感じますか？
q1 = st.radio(
    "Q1: ダーツを握るとき、指のどこにダーツの重さを感じますか？",
    ("A: 指の全体でしっかり握り、手のひらに近い方で重さを感じる",
      "B: 指の先端の方でバランスを取りながら、軽やかに感じる",
      "C: ダーツの真ん中あたりを指で探して、そこが一番しっくりくる")
)

# Q2: ダーツを投げた後、腕はどのように伸びていくのが一番自然だと感じますか？
q2 = st.radio(
    "Q2: ダーツを投げた後、腕はどのように伸びていくのが一番自然だと感じますか？",
    ("A: 狙った場所に向かって、腕をしっかり「押し出す」ように伸ばすのが自然",
      "B: 腕を「振り子」のように、力を抜いて自然に「振り抜く」のが自然",
      "C: 的に指を指すように、最後まで「まっすぐ」腕を伸ばしきるのが自然")
)

# Q3: ダーツがボードに刺さったとき、狙った場所からどうズレることが一番多いですか？
q3 = st.radio(
    "Q3: ダーツがボードに刺さったとき、狙った場所からどうズレることが一番多いですか？",
    ("A: 狙った場所より少し下に刺さることが多い",
      "B: 狙った場所より少し上に刺さることが多い",
      "C: 上下というより、左右にバラけることが多い")
)

st.markdown("---")

# --- ステップ3: ダーツフォーム動画をアップロードしてください ---
st.header("ステップ3: ダーツフォーム動画をアップロードしてください")
st.write("（**あなたが普段お使いのマイダーツで**3本～5本投げた動画がおすすめです。真横からの撮影がベスト）")
st.write("※マイダーツの重さや形状が診断に影響しますので、できるだけ普段お使いのダーツで撮影してください。")


uploaded_file = st.file_uploader("動画ファイルを選択してください", type=["mp4", "mov", "avi"])

# 動画を一時的に保存するディレクトリ
temp_dir = "./temp_video"
video_path = None # video_pathを初期化

if uploaded_file is not None:
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("動画がアップロードされました！")
    st.video(video_path) # アップロードされた動画をここで表示

st.markdown("---")


# --- ステップ3.5: 着弾点写真のアップロード ---
st.header("ステップ3.5: 🎯 着弾点写真のアップロード")
st.write("フォーム動画の撮影後、ダーツボードに刺さった状態の写真を撮り、アップロードしてください。")

# フォーム動画のアップローダーと同じ形式を使用
uploaded_photo = st.file_uploader("着弾点写真ファイルを選択してください", type=["jpg", "jpeg", "png"])

# 写真がアップロードされた場合の処理
if uploaded_photo is not None:
    # ユーザーにアップロードされた写真を表示
    st.image(uploaded_photo, caption="アップロードされた着弾点写真", use_column_width=True)
    
    st.info("💡 **【重要】着弾点分析のための注意点**")
    st.write("1. 写真はダーツボードの**真正面**から、歪みがないように撮影してください。")
    st.write("2. 着弾点分析の精度を高めるため、**全てのダーツ**が刺さった状態のものが理想です。")

st.markdown("---")

# --- 診断ボタンと結果表示 ---
if st.button("あなたの運命のダーツを診断！"):
    if uploaded_file is None:
        st.error("動画ファイルをアップロードしてください。")
    else:

        # 着弾点写真の保存ロジックをここに挿入
        if uploaded_photo is not None: # uploaded_photoがアップロードされているかチェック
            photo_dir = os.path.join(temp_dir, "photo")
            os.makedirs(photo_dir, exist_ok=True)
            photo_path = os.path.join(photo_dir, uploaded_photo.name)
            with open(photo_path, "wb") as f:
                f.write(uploaded_photo.getbuffer())
        else:
            photo_path = None # 写真がない場合はNoneを代入

        # --- ここからGoogle Gemini APIの設定と呼び出し ---
        try:
            # APIキーはStreamlitのSecretsから取得（推奨）
            if "GEMINI_API_KEY" not in st.secrets:
                 st.warning("⚠️ Google Gemini APIキーが設定されていません。AI診断は実行されません。")
                 st.info("Streamlit Cloud の場合は 'Secrets' に、ローカルテストの場合は `.[streamlit/secrets.toml]` ファイルに 'GEMINI_API_KEY' を設定してください。")
                 st.stop() # 処理を中断
            
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        except Exception as e:
            st.error(f"Gemini APIの設定中にエラーが発生しました: {e}")
            st.stop()

        
        with st.spinner("診断中... AIがあなたのフォームとダーツ感覚、好みを解析しています"):

             
             # --- プロンプトの組み立て ---
             prompt_parts = [
                "あなたはダーツの専門家であり、ダーツプレイヤーのパフォーマンス向上をサポートするAIです。",
                "ユーザーは自分のマイダーツを持っており、より深くダーツを理解し、次のステップに進みたいと考えています。",
                "以下のユーザーの現在のマイダーツ情報、ダーツを投げた際の感覚、そしてフォームの特性から、",
                "ユーザーに最適なダーツバレル（形状、重さ、カットの特徴）、または現在のセッティングからの改善点や次の試すべきダーツの方向性を、具体的な理由とともに提案してください。",
                "回答は、経験者にも納得感があり、かつ親しみやすい言葉遣いでお願いします。",
                "---ユーザーの現在のマイダーツ情報---",
                f"バレルの形状: {current_barrel_type}",
                f"バレルの重さ: {current_barrel_weight}g",
                f"シャフトの長さ: {current_shaft_length}",
                f"フライトの形状: {current_flight_shape}",
                "---ユーザーのダーツ感覚---",
                f"Q1（握る感覚）: {q1}",
                f"Q2（腕の伸び方）: {q2}",
                f"Q3（ダーツのズレ）: {q3}",
             ]

            # 予算に関する情報もプロンプトに含める
             budget_info = ""
             if q0_2.startswith("A:"): budget_info = "予算は5,000円くらいで、最初は安く始めたい。"
             elif q0_2.startswith("B:"): budget_info = "予算は5,000円〜15,000円で、長く使えるものが欲しい。"            
             elif q0_2.startswith("C:"): budget_info = "予算は15,000円以上で、デザインも性能も妥協したくない。"
             else: budget_info = "予算は診断結果を見てから決めたい。"
             prompt_parts.append(f"---ユーザーの予算に関する希望---\n{budget_info}")
            
            # --- 着弾点データの前処理とプロンプトへの追加 (写真アップロードに切り替え済み) ---
             if photo_path is not None:
                # ユーザーが着弾点写真を提供したことをAIに伝える
                prompt_parts.append(f"\n---着弾点写真による分析---")
                prompt_parts.append(f"ユーザーはダーツボードの写真を提供しました。この写真から、着弾が全体的にどこに偏っているか、散らばり具合はどうかを視覚的に想像し、診断に活かしてください。")
                # 💡 将来的にこの場所でOpenCVやVision AIを使って画像を解析するロジックを実装します。

            
            # --- 復元した動画解析の処理 (MediaPipe) ---
             mp_pose = mp.solutions.pose
             mp_drawing = mp.solutions.drawing_utils

             st.subheader(f"📊 詳細フォーム分析（試作版：あなたの{dominant_arm}の肘の角度）")
             st.write(f"動画からあなたの{dominant_arm}の肘の角度を抽出し、その推移をグラフで表示します。")

             # 動画処理の実行
             elbow_angles, over_path, cap, out = process_video_for_analysis(video_path, dominant_arm, mp_pose, mp_drawing)

             # --- フォーム解析結果の表示ロジック ---
             if elbow_angles: # 解析が成功した場合のみ表示
                
                # 1. グラフの表示
                fig, ax = plt.subplots()
                ax.plot(elbow_angles)
                ax.set_title("肘の角度の時系列変化")
                ax.set_xlabel("フレーム")
                ax.set_ylabel("角度 (度)")
                st.pyplot(fig) 
                plt.close(fig)

                # 2. 数値結果の表示とプロンプトへの追加
                avg_angle = np.mean(elbow_angles)
                std_dev_angle = np.std(elbow_angles)
                
                st.write(f"**平均肘角度:** {avg_angle:.2f}度")
                st.write(f"**肘角度の標準偏差（ブレ）:** {std_dev_angle:.2f}度")
                st.info("※肘の角度のブレが小さいほど、安定したリリースができていることを示します。")
                
                prompt_parts.append(f"\n---フォーム分析（数値データ）---")
                prompt_parts.append(f"平均肘角度: {avg_angle:.2f}度")
                prompt_parts.append(f"肘角度の標準偏差（ブレ）: {std_dev_angle:.2f}度")
                
                # 3. オーバーレイ動画の表示
                if over_path and os.path.exists(over_path):
                    st.subheader("👀 あなたのフォーム（骨格オーバーレイ）")
                    st.video(over_path)
                 
             else:
                 # ランドマークが検出できなかった場合
                 st.warning("動画からフォームのランドマークを検出できませんでした。撮影アングルや照明、人物の写り方をご確認ください。")


             # --- Gemini AIへのプロンプト作成と呼び出し ---
             model = genai.GenerativeModel('gemini-2.5-pro')
             generation_config = genai.GenerationConfig(temperature=0.7) 

             try:
                # プロンプトを結合してAIに送信
                response = model.generate_content("\n".join(prompt_parts), generation_config=generation_config)
                
                st.subheader("🎁 AIが提案するあなたの「運命のダーツ」はこれ！")
                st.success(response.text) # AIが生成したテキストをそのまま表示

             except Exception as e:
                st.error(f"AI診断中にエラーが発生しました。エラー詳細: {e}")
                
            # --- 診断結果の結びの文章 ---
             st.markdown("---")
             st.write("### 🎯 診断結果を参考に、次の一歩を踏み出そう！")
             st.write("この診断は、AIがあなたの**深い感覚とフォームの特性**から推測したものです。")
             st.write("最終的には、実際にダーツショップなどで**専門スタッフに相談**し、**様々なダーツを『試投』**して、あなたの手に最も馴染む一本を見つけることが重要です。")
             st.write("この診断結果をヒントに、ぜひあなたのダーツを次のレベルへと進化させてくださいね！")
            
# --- 最終的な一時ファイルの削除ロジック (診断ボタンの外側で定義) ---
# ※ 診断ボタンが押されたかどうかに関わらず、ファイルを解放・削除するために実行。
if video_path and os.path.exists(temp_dir): # temp_dirが存在する場合に実行
    # 処理後に cap と out が閉じられていない場合のために del を実行
    try:
        import gc # ガベージコレクションをインポート
        if 'cap' in locals() and cap is not None:
            cap.release()
            del cap
        if 'out' in locals() and out is not None:
            out.release()
            del out
        gc.collect() # 強制ガベージコレクション
        
        # ディレクトリごと削除
        shutil.rmtree(temp_dir) 
        
    except Exception as e:
        # WinError 5 アクセス拒否エラーが出やすい場所。警告にとどめる。
        st.warning(f"一時ディレクトリの削除中にエラーが発生しました: {e}。手動で '{temp_dir}' フォルダを削除してください。")