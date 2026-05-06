import streamlit as st
from PIL import Image
import io
import requests
from openai import OpenAI

# ========================
# LOAD SECRETS (Streamlit Cloud)
# ========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========================
# UI
# ========================
st.set_page_config(page_title="Corneal Ulcer AI v3", layout="centered")

st.title("👁️ ระบบคัดกรองแผลกระจกตา KU KPS Infirmary")
st.markdown("ใช้สำหรับ คัดกรองเบื้องต้น + telemedicine")

# ========================
# INPUT: IMAGE
# ========================
uploaded_file = st.file_uploader("📸 อัปโหลดภาพตา", type=["jpg", "png", "jpeg"])

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
NYCKEL_CLIENT_ID     = st.secrets.get("NYCKEL_CLIENT_ID", "rbobnmmaxplm0qamqy3rabxzn9o4pu72")
NYCKEL_CLIENT_SECRET = st.secrets.get("NYCKEL_CLIENT_SECRET", "0n4efo30nrdsomchfih0u71gdpndxhl2hjmdlw71u24ix4b0x6o6d68sln0p0p25")

def get_nyckel_token():
    """ขอ access token ใหม่ด้วย OAuth2 Client Credentials Flow"""
    try:
        response = requests.post(
            "https://www.nyckel.com/connect/token",
            data={
                "grant_type":    "client_credentials",
                "client_id":     NYCKEL_CLIENT_ID,
                "client_secret": NYCKEL_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        return None

def analyze_image_nyckel(image):
    # Step 1: ขอ token
    token = get_nyckel_token()
    if not token:
        return {"error": "ไม่สามารถขอ access token จาก Nyckel ได้"}

    # Step 2: เตรียมภาพเป็น bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    # Step 3: ส่งเป็น multipart/form-data
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

    # Step 4: ตรวจ response
    if "labelName" not in result:
        return {"error": result.get("message", f"Unexpected response: {result}")}

    return result

# ========================
# SYMPTOM SCORE
# ========================
def calculate_symptom_score():
    score = 0
    score += pain * 2
    if photophobia: score += 10
    if redness: score += 5
    if vision_loss: score += 15
    if discharge: score += 5
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
def gpt_summary(ai_result, symptom_score, risk_level):
    if "error" in ai_result:
        ai_text = f"AI ล้มเหลว: {ai_result['error']} (ห้ามสรุปจากภาพ)"
    else:
        ai_text = f"AI พบ: {ai_result}"

    prompt = f"""
    วิเคราะห์ข้อมูล:

    - ผล AI: {ai_text}
    - คะแนนอาการ: {symptom_score}
    - ระดับความเสี่ยงที่ประเมินแล้ว: {risk_level}

    กฎสำคัญ:
    - ถ้า AI ล้มเหลว → ห้ามสรุปหรืออ้างอิงจากภาพเด็ดขาด
    - ให้ใช้ symptom score เป็นหลักในการอธิบาย
    - ต้องสอดคล้องกับระดับความเสี่ยงที่ให้ไว้

    ตอบเป็นภาษาไทย:
    1. เหตุผลของระดับความเสี่ยง ({risk_level})
    2. อาการที่น่าเป็นห่วง (ถ้ามี)
    3. คำแนะนำ (รักษาเบื้องต้น / ส่งต่อด่วน)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content

# ========================
# MAIN
# ========================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์"):
        with st.spinner("กำลังวิเคราะห์..."):

            # AI Image
            ai_result = analyze_image_nyckel(image)

            # Symptom
            symptom_score = calculate_symptom_score()

            # ถ้า AI พัง → ไม่เอาไปคำนวณ
            if "error" in ai_result:
                ai_conf = 0
                ai_weight = 0
            else:
                ai_conf = ai_result.get("confidence", 0)
                ai_weight = 100

            # คำนวณ final score
            final_score = (ai_conf * ai_weight) + symptom_score

            # Risk level จาก clinical logic
            risk_level = calculate_risk_level(ai_result, final_score, symptom_score)

            # GPT Summary
            summary = gpt_summary(ai_result, symptom_score, risk_level)

        # ========================
        # OUTPUT
        # ========================
        st.subheader("📊 ผลลัพธ์")

        st.write("### 🤖 AI ตรวจภาพ")
        if "error" in ai_result:
            st.error(f"❌ AI Image Error: {ai_result['error']}")
            st.info("ℹ️ ระบบจะประเมินจากอาการเป็นหลัก")
        else:
            st.json(ai_result)

        st.write("### 🧾 คะแนนอาการ")
        st.write(symptom_score)

        st.write("### 📈 คะแนนรวม (Hybrid Risk)")
        st.metric("Risk Score", round(final_score, 2))

        # Risk level display
        if risk_level == "สูง":
            st.error("🔴 ความเสี่ยงสูง (ควรส่งต่อด่วน)")
        elif risk_level == "ปานกลาง":
            st.warning("🟠 ความเสี่ยงปานกลาง")
        else:
            st.success("🟢 ความเสี่ยงต่ำ")

        st.write("### 🧠 สรุปโดย AI")
        st.write(summary)

        st.warning("⚠️ ใช้เพื่อคัดกรองเท่านั้น ไม่ใช่วินิจฉัย")
