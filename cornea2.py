import streamlit as st
from PIL import Image
import io
import os
import tempfile
import requests
import cv2
from openai import OpenAI

# ========================
# LOAD SECRETS (Streamlit Cloud)
# ========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========================
# UI
# ========================
st.set_page_config(page_title="Corneal Ulcer AI v4", layout="centered")

st.title("👁️ ระบบคัดกรองแผลกระจกตา KU KPS Infirmary")
st.markdown("ใช้สำหรับคัดกรองเบื้องต้น + telemedicine")

# ========================
# INPUT: IMAGE OR VIDEO
# ========================
st.subheader("📸 เลือกชนิดไฟล์")
media_type = st.radio(
    "ต้องการอัปโหลดแบบใด",
    ["ภาพนิ่ง", "วิดีโอ"],
    horizontal=True,
)

uploaded_file = None
image = None
video_frames = []

if media_type == "ภาพนิ่ง":
    uploaded_file = st.file_uploader(
        "📸 อัปโหลดภาพตา",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

else:
    uploaded_file = st.file_uploader(
        "🎥 อัปโหลดวิดีโอตา",
        type=["mp4", "mov", "avi", "m4v"],
        key="video_upload",
    )
    if uploaded_file:
        st.video(uploaded_file)
        st.info("ระบบจะดึงเฟรมจากวิดีโอเพื่อนำไปวิเคราะห์ด้วย Nyckel")

# ========================
# INPUT: SYMPTOMS
# ========================
st.subheader("🧾 อาการผู้ป่วย")

pain = st.slider("ปวดตา", 0, 10, 3)
photophobia = st.checkbox("แพ้แสง")
redness = st.checkbox("ตาแดง")
vision_loss = st.checkbox("ตามัว")
discharge = st.checkbox("มีขี้ตา")

# ========================
# NYCKEL OAuth2 — Client Credentials Flow
# ========================
NYCKEL_CLIENT_ID = st.secrets.get("NYCKEL_CLIENT_ID", "")
NYCKEL_CLIENT_SECRET = st.secrets.get("NYCKEL_CLIENT_SECRET", "")
# Optional: ใช้เฉพาะกรณีที่อาจารย์มี access token แบบชั่วคราวอยู่แล้ว
# โดยทั่วไปไม่จำเป็น เพราะระบบจะขอ token อัตโนมัติจาก CLIENT_ID/CLIENT_SECRET
NYCKEL_ACCESS_TOKEN = st.secrets.get("NYCKEL_ACCESS_TOKEN", "")


def get_nyckel_token():
    """ขอ access token อัตโนมัติด้วย OAuth2 Client Credentials Flow"""
    if NYCKEL_ACCESS_TOKEN:
        return NYCKEL_ACCESS_TOKEN

    if not NYCKEL_CLIENT_ID or not NYCKEL_CLIENT_SECRET:
        return None

    try:
        response = requests.post(
            "https://www.nyckel.com/connect/token",
            data={
                "grant_type": "client_credentials",
                "client_id": NYCKEL_CLIENT_ID,
                "client_secret": NYCKEL_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        token_data = response.json()
        return token_data.get("access_token")
    except Exception:
        return None


def analyze_image_nyckel(image_obj):
    """ส่งภาพ 1 ภาพไปยัง Nyckel"""
    token = get_nyckel_token()
    if not token:
        return {
            "error": (
                "ไม่สามารถขอ access token จาก Nyckel ได้ — "
                "กรุณาตรวจ Streamlit Secrets ว่ามี NYCKEL_CLIENT_ID และ "
                "NYCKEL_CLIENT_SECRET ถูกต้อง หรือใส่ NYCKEL_ACCESS_TOKEN ชั่วคราว"
            )
        }

    buffered = io.BytesIO()
    image_obj.save(buffered, format="JPEG")
    buffered.seek(0)

    try:
        response = requests.post(
            "https://www.nyckel.com/v1/functions/corneal-ulcer/invoke",
            files={"file": ("image.jpg", buffered, "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        result = response.json()
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

    if "labelName" not in result:
        return {"error": result.get("message", f"Unexpected response: {result}")}

    return result


def extract_video_frames(uploaded_video, max_frames=5):
    """
    ดึงเฟรมจากวิดีโอแบบกระจายช่วงเวลา
    คืนค่าเป็น list ของ PIL Image
    """
    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_video.getbuffer())
        tmp_path = tmp_file.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return []

        # เลือกเฟรมกระจาย เช่น ต้น-กลาง-ท้าย เพื่อลดภาระ API
        frame_indices = []
        for i in range(max_frames):
            idx = int((i + 1) * total_frames / (max_frames + 1))
            frame_indices.append(idx)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb).convert("RGB"))

        cap.release()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return frames


def analyze_video_nyckel(frames):
    """
    วิเคราะห์หลายเฟรม แล้วเลือกผลที่ confidence สูงสุด
    """
    results = []

    for i, frame in enumerate(frames, start=1):
        result = analyze_image_nyckel(frame)
        result["frame_no"] = i
        results.append(result)

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return {
            "error": "วิเคราะห์วิดีโอไม่สำเร็จทุกเฟรม",
            "frame_results": results,
        }

    best_result = max(valid_results, key=lambda x: x.get("confidence", 0))
    best_result["frame_results"] = results
    best_result["analyzed_frames"] = len(results)
    return best_result


# ========================
# SYMPTOM SCORE
# ========================
def calculate_symptom_score():
    score = 0
    score += pain * 2
    if photophobia:
        score += 10
    if redness:
        score += 5
    if vision_loss:
        score += 15
    if discharge:
        score += 5
    return score


# ========================
# RISK LEVEL
# ========================
def calculate_risk_level(ai_result, final_score, symptom_score):
    if "error" in ai_result:
        if symptom_score > 40:
            return "สูง"
        elif symptom_score > 20:
            return "ปานกลาง"
        else:
            return "ต่ำ"
    else:
        if final_score > 120:
            return "สูง"
        elif final_score > 70:
            return "ปานกลาง"
        else:
            return "ต่ำ"


# ========================
# GPT SUMMARY
# ========================
def gpt_summary(ai_result, symptom_score, risk_level, media_type):
    if "error" in ai_result:
        ai_text = f"AI ล้มเหลว: {ai_result['error']} (ห้ามสรุปจากภาพหรือวิดีโอ)"
    else:
        ai_text = f"AI พบจาก{media_type}: {ai_result}"

    prompt = f"""
    วิเคราะห์ข้อมูล:

    - ชนิดไฟล์: {media_type}
    - ผล AI: {ai_text}
    - คะแนนอาการ: {symptom_score}
    - ระดับความเสี่ยงที่ประเมินแล้ว: {risk_level}

    กฎสำคัญ:
    - ถ้า AI ล้มเหลว → ห้ามสรุปหรืออ้างอิงจากภาพ/วิดีโอเด็ดขาด
    - ให้ใช้ symptom score เป็นหลักในการอธิบาย
    - ต้องสอดคล้องกับระดับความเสี่ยงที่ให้ไว้
    - เป็นระบบคัดกรอง ไม่ใช่การวินิจฉัยยืนยัน

    ตอบเป็นภาษาไทย:
    1. เหตุผลของระดับความเสี่ยง ({risk_level})
    2. อาการที่น่าเป็นห่วง (ถ้ามี)
    3. คำแนะนำ (รักษาเบื้องต้น / ส่งต่อด่วน)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return response.choices[0].message.content


# ========================
# MAIN
# ========================
if uploaded_file:
    if st.button("🔍 วิเคราะห์"):
        with st.spinner("กำลังวิเคราะห์..."):
            symptom_score = calculate_symptom_score()

            if media_type == "ภาพนิ่ง":
                ai_result = analyze_image_nyckel(image)
                preview_frames = []
            else:
                preview_frames = extract_video_frames(uploaded_file, max_frames=5)
                if not preview_frames:
                    ai_result = {"error": "ไม่สามารถดึงเฟรมจากวิดีโอได้"}
                else:
                    ai_result = analyze_video_nyckel(preview_frames)

            if "error" in ai_result:
                ai_conf = 0
                ai_weight = 0
            else:
                ai_conf = ai_result.get("confidence", 0)
                ai_weight = 100

            final_score = (ai_conf * ai_weight) + symptom_score
            risk_level = calculate_risk_level(ai_result, final_score, symptom_score)
            summary = gpt_summary(ai_result, symptom_score, risk_level, media_type)

        # ========================
        # OUTPUT
        # ========================
        st.subheader("📊 ผลลัพธ์")

        if media_type == "วิดีโอ" and preview_frames:
            st.write("### 🎞️ เฟรมที่ระบบดึงมาวิเคราะห์")
            st.image(
                preview_frames,
                caption=[f"Frame {i}" for i in range(1, len(preview_frames) + 1)],
                use_container_width=True,
            )

        st.write("### 🤖 AI ตรวจภาพ/วิดีโอ")
        if "error" in ai_result:
            st.error(f"❌ AI Error: {ai_result['error']}")
            st.info("ℹ️ ระบบจะประเมินจากอาการเป็นหลัก")
            if "frame_results" in ai_result:
                st.json(ai_result["frame_results"])
        else:
            st.json(ai_result)

        st.write("### 🧾 คะแนนอาการ")
        st.write(symptom_score)

        st.write("### 📈 คะแนนรวม (Hybrid Risk)")
        st.metric("Risk Score", round(final_score, 2))

        if risk_level == "สูง":
            st.error("🔴 ความเสี่ยงสูง (ควรส่งต่อด่วน)")
        elif risk_level == "ปานกลาง":
            st.warning("🟠 ความเสี่ยงปานกลาง")
        else:
            st.success("🟢 ความเสี่ยงต่ำ")

        st.write("### 🧠 สรุปโดย AI")
        st.write(summary)

        st.warning("⚠️ ใช้เพื่อคัดกรองเท่านั้น ไม่ใช่วินิจฉัย")
else:
    st.info("กรุณาเลือกและอัปโหลดภาพนิ่งหรือวิดีโอก่อนเริ่มวิเคราะห์")

