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

# 記録用のセッションステートを初期化
if 'hit_points' not in st.session_state:
    st.session_state.hit_points = []
    
# リセット回数カウンター（ウィジェットのキーを更新し、リセット問題を解決するために使用）
if 'reset_count' not in st.session_state:
    st.session_state.reset_count = 0

# =========================================================
# 3. Matplotlib描画とPIL Image変換を行うヘルパー関数
# =========================================================
def create_drawable_darts_board(img_path, hit_points):
    """ダーツボード画像に記録点を描画し、クリック可能なPIL Imageオブジェクトを返す"""
    try:
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        
        if hit_points:
            x_coords = [p[0] for p in hit_points]
            y_coords = [p[1] for p in hit_points]
            ax.scatter(x_coords, y_coords, color='red', s=100, alpha=0.8, edgecolors='black', linewidths=1.5)
        
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0) 
        ax.axis('off') 
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_pil = Image.open(buf)
        
        plt.close(fig) 
        return img_pil
        
    except Exception as e:
        st.error(f"描画関数内で致命的なエラーが発生しました: {e}")
        return None

# =========================================================
# 4. フォーム解析のためのヘルパー関数 (復元・追加)
# =========================================================
def calculate_angle(a, b, c):
    """3つのランドマークから角度を計算する"""
    a = np.array(a)  # 最初の点 (例: 肩)
    b = np.array(b)  # 中心の点 (例: 肘)
    c = np.array(c)  # 最後の点 (例: 手首)
    
    # ベクトルBAとベクトルBCの角度を計算
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_video_for_analysis(video_path, dominant_arm, mp_pose, mp_drawing):
    """動画を解析し、肘の角度リストとオーバーレイ動画パスを返す"""
    elbow_angles = []
    # オーバーレイ動画の保存先
    over_path = os.path.join(os.path.dirname(video_path), "overlay_output.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return elbow_angles, None, cap, None

    # 動画の基本情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # オーバーレイ動画を書き出すための設定（コーデックに注意）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Streamlitと互換性の高いコーデック
    out = cv2.VideoWriter(over_path, fourcc, fps, (width, height))
    
    # MediaPipe Poseモデルの初期化
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        pbar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR画像をRGBに変換（MediaPipe処理のため）
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 姿勢推定の実行
            results = pose.process(image)
            
            # RGB画像をBGRに戻す（OpenCV表示/保存のため）
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # ランドマークが検出された場合のみ処理
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 利き腕に応じてランドマークを設定
                    shoulder_landmark = mp_pose.PoseLandmark.RIGHT_SHOULDER if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_SHOULDER
                    elbow_landmark = mp_pose.PoseLandmark.RIGHT_ELBOW if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_ELBOW
                    wrist_landmark = mp_pose.PoseLandmark.RIGHT_WRIST if dominant_arm == "右利き" else mp_pose.PoseLandmark.LEFT_WRIST
                    
                    shoulder = [landmarks[shoulder_landmark.value].x, landmarks[shoulder_landmark.value].y]
                    elbow = [landmarks[elbow_landmark.value].x, landmarks[elbow_landmark.value].y]
                    wrist = [landmarks[wrist_landmark.value].x, landmarks[wrist_landmark.value].y]
                        
                    # 角度を計算し、リストに追加
                    angle_deg = calculate_angle(shoulder, elbow, wrist)
                    elbow_angles.append(angle_deg)
                    
                    # 骨格を描画
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                                
                except Exception:
                    # 角度計算エラーは無視して処理を続行
                    pass 
            
            # オーバーレイ動画として書き出し
            out.write(image)
            
            frame_count += 1
            # プログレスバーの更新
            pbar.progress(min(frame_count / (total_frames + 1), 1.0))
        
        pbar.empty() # プログレスバーを削除
        
    return elbow_angles, over_path, cap, out

# --- アプリのメイン表示 ---
st.title("🎯 あなたのダーツ、次なる進化へ！【マイダーツ深掘り診断】")
st.write("今のマイダーツに、もう少しフィット感が欲しいと思いませんか？")
st.write("この診断では、**あなたが普段使っているマイダーツの情報**と**投げ方の感覚**を深掘りし、**フォームの特性**、そして**予算**も考慮して、あなたに最適な『次の一本』や『セッティングのヒント』を提案します！")
st.write("さあ、あなたのダーツを次のレベルへと進化させましょう！")

st.markdown("---")

# --- カスタムCSSの定義 ---
st.markdown("""
<style>
/* ラジオボタンのスタイル */
.stRadio > label {
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 5px;
}
.stRadio div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 20px;
}
.stApp {
    background-color: #191919; /* 暗い背景 */
    color: #E0E0E0; /* 全体の文字色 */

        /* 🌟【追加】背景画像の設定 🌟 */
    background-image: url("https://raw.githubusercontent.com/k024c1009-ui/darts_app/refs/heads/main/.gitignore"); 
    background-size: cover; /* 画面全体に画像を拡大/縮小して表示 */
    background-attachment: fixed; /* スクロールしても背景を固定 */
    background-position: center;
}
/* 全ての標準テキスト（pタグなど）も統一 */
p, li {
    color: #E0E0E0 !important;
}

/* 🌟【追加】ラジオボタンの文字色を強制的に明るくする 🌟 */
.stRadio label {
    color: white !important; /* 選択肢の文字色を強制的に白に */
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 5px;
}

/* ヘッダーの色をモダンなアクセントカラーに */
h1 {
    color: #00FFFF; /* ターコイズブルー */
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.5); /* ネオン効果 */
}
h2, h3 {
    color: #00FFFF;
}

/* 診断ボタンのカスタマイズ */
.stButton>button {
    background-color: #00FFFF; /* 背景色 */
    color: #191919; /* 文字色 */
    border: 2px solid #00FFFF;
    border-radius: 8px;
    font-weight: bold;
    padding: 10px 20px;
    transition: all 0.2s;
}

/* ホバー時の効果（任意） */
.stButton>button:hover {
    background-color: #191919;
    color: #00FFFF;
    box-shadow: 0 0 10px #00FFFF;
}

</style>
""", unsafe_allow_html=True)


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

    # TODO: 今後、ここで写真のパスを一時保存するロジックを追加する

st.markdown("---")

# 元の着弾点手動入力UI（streamlit_image_coordinatesの部分）は完全に削除します。
# ----------------------------------------------------------------------------------
# (元の st.session_state.hit_points や streamlit_image_coordinates のロジックは全て削除)
# ----------------------------------------------------------------------------------

# --- 診断ボタンと結果表示 ---
if st.button("あなたの運命のダーツを診断！"):
    if uploaded_file is None:
        st.error("動画ファイルをアップロードしてください。")
    else:

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

            if photo_path is not None:
             
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
            
            # --- 着弾点データの前処理とプロンプトへの追加 ---
             if st.session_state.hit_points:
                # 記録された座標をNumPy配列に変換
                points = np.array(st.session_state.hit_points)
                
                # X座標とY座標の平均（中心）と標準偏差（散らばり）を計算
                mean_x = np.mean(points[:, 0]) 
                mean_y = np.mean(points[:, 1]) 
                std_x = np.std(points[:, 0])   
                std_y = np.std(points[:, 1])   
                
                # 計算結果をプロンプトに追加
                prompt_parts.append(f"\n---着弾点分析の結果---")
                prompt_parts.append(f"着弾点の中心 (X, Y): ({mean_x:.2f}, {mean_y:.2f})")
                prompt_parts.append(f"X軸の散らばり（ブレの目安）: {std_x:.2f} ピクセル")
                prompt_parts.append(f"Y軸の散らばり（ブレの目安）: {std_y:.2f} ピクセル")
                prompt_parts.append("※ X座標は左ほど小さく、Y座標は上ほど小さい画像座標です。")
                
                # 簡易的な定性分析（AIが判断しやすくするためのヒント）
                if std_x > 50 or std_y > 50: 
                    prompt_parts.append("全体の散らばり（標準偏差）が大きいため、リリースが非常に不安定です。")
                if mean_y > 350: # 例としてボード下部をY>350と仮定
                    prompt_parts.append("着弾中心がボードの下方に大きく偏っており、ダーツが失速しやすい傾向です。")
            
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
            
# --- 最終的な一時ファイルの削除ロジック (WinError 5対策済み) ---
# ※ 診断ボタンが押されたかどうかに関わらず、ファイルを解放・削除するために実行。
if video_path and os.path.exists(temp_dir): # temp_dirが存在する場合に実行
    # 処理後に cap と out が閉じられていない場合のために del を実行
    try:
        if 'cap' in locals() and cap is not None:
            cap.release()
            del cap
        if 'out' in locals() and out is not None:
            out.release()
            del out
            
        # ディレクトリごと削除
        shutil.rmtree(temp_dir) 
        # st.success(f"一時ディレクトリ '{temp_dir}' を正常に削除しました。") # デバッグ用
        
    except Exception as e:
        # WinError 5 アクセス拒否エラーが出やすい場所。警告にとどめる。
        st.warning(f"一時ディレクトリの削除中にエラーが発生しました: {e}。手動で '{temp_dir}' フォルダを削除してください。")