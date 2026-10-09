from datetime import datetime
import sqlite3
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PCOS AI Clinical Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# تهيئة قاعدة البيانات المحلية لحفظ السجلات الطبية
def init_db():
  conn = sqlite3.connect("pcos_records.db")
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            insulin REAL,
            testosterone REAL,
            tsh REAL,
            prolactin REAL,
            lh REAL,
            fsh REAL,
            dheas REAL,
            shbg REAL,
            hba1c REAL,
            vitD REAL,
            report TEXT,
            has_result INTEGER
        )
    """)
  conn.commit()
  conn.close()


init_db()


class PatientInput(BaseModel):
  insulin: float = 0.0
  testosterone: float = 0.0
  tsh: float = 0.0
  prolactin: float = 0.0
  lh: float = 0.0
  fsh: float = 0.0
  dheas: float = 0.0
  shbg: float = 0.0
  hba1c: float = 0.0
  vitD: float = 0.0


class PCOSTheAIModel:

  def analyze_case(self, data: PatientInput) -> Dict[str, Any]:
    risk_scores = {
        "insulin_resistance": 0,
        "hyperandrogenism_ovarian": 0,
        "adrenal_stress": 0,
        "thyroid_prolactin": 0,
        "micronutrient_deficiency": 0,
    }

    findings = []
    protocols = []
    diet_recommendations = []
    exercise_recommendations = []

    # تتبع ما إذا كانت المريضة قد أدخلت أي قيمة حقيقية أكبر من الصفر
    entered_any_value = False

    # 1. تحليل محور مقاومة الإنسولين (يعمل إذا تم إدخال إنسولين أو تراكمي)
    if data.insulin > 0.0 or data.hba1c > 0.0:
      entered_any_value = True
      if data.insulin > 10.0 or data.hba1c >= 5.7:
        risk_scores["insulin_resistance"] += 2
        findings.append(
            "• نموذج الذكاء الاصطناعي رصد مؤشرات واضحة لسيادة مقاومة الإنسولين"
            " الأيضية من الفحوصات المتاحة."
        )
        protocols.append(
            "• بروتوكول الأيض: مكملات ميو-إينوسيتول وحمض الألفا ليبويك."
        )
        diet_recommendations.append(
            "🥗 الأغذية المناسبة لمقاومة الإنسولين: التركيز على الخضروات الورقية،"
            " الدهون الصحية (أفوكادو، زيت الزيتون)، والبروتينات. وتجنب"
            " السكريات والدقيق الأبيض تماماً."
        )
        exercise_recommendations.append(
            "🏃‍♀️ التمارين المناسبة: المشي السريع لمدة 30 دقيقة بعد الوجبات"
            " لتحسين حساسية الخلايا للإنسولين."
        )

    # 2. تحليل محور المبايض والأندروجينات
    if data.testosterone > 0.0 or data.lh > 0.0 or data.fsh > 0.0:
      entered_any_value = True
      if data.testosterone > 0.7 or (data.lh > 0 and data.lh > 10.0):
        risk_scores["hyperandrogenism_ovarian"] += 2
        findings.append(
            "• نموذج الذكاء الاصطناعي رصد فرط أندروجين وارتفاع في هرمونات المبيض"
            " من الفحوصات المدخلة."
        )
        protocols.append(
            "• بروتوكول التوازن الهرموني: شاي النعناع البلدي ومكملات الساو"
            " بالميتو لخفض الهرمونات الذكرية."
        )
        diet_recommendations.append(
            "🥗 الأغذية المخفضة للأندروجين: تناول بذور الكتان والشاي الأخضر"
            " للحد من ارتفاع الهرمونات الذكرية."
        )

    # 3. تحليل المحور الكظري (الإجهاد والتوتر)
    if data.dheas > 0.0:
      entered_any_value = True
      if data.dheas > 340.0:
        risk_scores["adrenal_stress"] += 2
        findings.append(
            "• نموذج الذكاء الاصطناعي استنتج وجود إجهاد كظري (أدرينالي) من فحص"
            " DHEA-S المدخل."
        )
        protocols.append(
            "• بروتوكول الغدة الكظرية: مغنيسيوم جلايسينات وأشواغاندا لخفض الكورتيزول."
        )
        exercise_recommendations.append(
            "🧘‍♀️ التمارين المناسبة: تجنب التمارين القاسية المنهكة، والتركيز على"
            " اليوغا والتنفس العميق لخفض التوتر."
        )

    # 4. الغدة الدرقية وهرمون الحليب
    if data.tsh > 0.0 or data.prolactin > 0.0:
      entered_any_value = True
      if data.tsh > 4.0 or (0 < data.tsh < 0.4) or data.prolactin > 25.0:
        risk_scores["thyroid_prolactin"] += 2
        findings.append(
            "• نموذج الذكاء الاصطناعي رصد تداخلاً محتملاً في محاور الغدة الدرقية"
            " أو البرولاكتين."
        )
        protocols.append(
            "• بروتوكول غددي: استشارة أخصائي غدد صماء لمتابعة استقرار TSH وهرمون"
            " الحليب."
        )

    # 5. نقص المغذيات وفيتامين د
    if data.vitD > 0.0:
      entered_any_value = True
      if data.vitD < 30.0:
        risk_scores["micronutrient_deficiency"] += 1
        findings.append(
            "• نموذج الذكاء الاصطناعي حدد نقصاً في مخزون فيتامين د3 من القيمة"
            " المدخلة."
        )
        protocols.append(
            "• بروتوكول الدعم الحيوي: جرعة علاجية من فيتامين د3 مع كيه 2 (K2)."
        )

    has_issue = len(findings) > 0

    # إذا لم تقم المريضة بإدخال أي قيمة نهائياً
    if not entered_any_value:
      report = (
          "⚠️ يرجى إدخال قيمة فحص واحد على الأقل ليقوم المحرك السريري بتحليله"
          " وإصدار البروتوكول."
      )
      has_issue = False
    elif not has_issue:
      # إذا أدخلت فحوصات وكانت قيمها سليمة وطبيعية ضمن النطاق
      diet_text = (
          "🥗 التوجيه الغذائي: نظام غذائي متوازن غني بالألياف والبروتينات، مع"
          " تقليل السكريات المصنعة."
      )
      exercise_text = (
          "🏃‍♀️ التمارين: الحفاظ على نشاط بدني منتظم (مشي أو تمارين خفيفة 3-4"
          " مرات أسبوعياً)."
      )
      report = (
          "🤖 تقرير الذكاء الاصطناعي السريري (الفحوصات المتاحة):\n\n"
          "✅ بناءً على تحليل المؤشرات المدخلة، القيم المذكورة تقع ضمن النطاق"
          " المستقر ولا توجد مؤشرات شاذة ظاهرة.\n\n"
          + diet_text
          + "\n\n"
          + exercise_text
      )
    else:
      # إذا تم رصد مشاكل في الفحوصات الجزئية المدخلة
      diet_text = (
          "\n".join(diet_recommendations)
          if diet_recommendations
          else "🥗 التوجيه الغذائي: التركيز على الأغذية الطازجة المتوازنة."
      )
      exercise_text = (
          "\n".join(exercise_recommendations)
          if exercise_recommendations
          else "🏃‍♀️ التمارين: الحفاظ على حركة نشطة وممارسة المشي المنتظم."
      )

      report = (
          "🤖 **تقرير الذكاء الاصطناعي والتحليل الجذري (للفحوصات المتاحة):**\n\n"
          + "\n".join(findings)
          + "\n\n"
          + "📋 **البروتوكول العلاجي الذكي المخصص:**\n\n"
          + "\n".join(protocols)
          + "\n\n"
          + diet_text
          + "\n\n"
          + exercise_text
      )

    # حفظ السجل في قاعدة البيانات
    try:
      conn = sqlite3.connect("pcos_records.db")
      cursor = conn.cursor()
      cursor.execute(
          """
                INSERT INTO patient_records (timestamp, insulin, testosterone, tsh, prolactin, lh, fsh, dheas, shbg, hba1c, vitD, report, has_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
          (
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              data.insulin,
              data.testosterone,
              data.tsh,
              data.prolactin,
              data.lh,
              data.fsh,
              data.dheas,
              data.shbg,
              data.hba1c,
              data.vitD,
              report,
              1 if has_issue else 0,
          ),
      )
      conn.commit()
      conn.close()
    except Exception as e:
      print(f"Database error: {e}")

    return {
        "hasResult": has_issue,
        "report": report,
        "ai_confidence_scores": risk_scores,
    }


ai_model = PCOSTheAIModel()


@app.post("/api/analyze")
def analyze_patient(data: PatientInput):
  return ai_model.analyze_case(data)


# مسار جلب السجلات والتقارير التاريخية للمريضة
@app.get("/api/history")
def get_patient_history():
  try:
    conn = sqlite3.connect("pcos_records.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_records ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
      history.append({
          "id": row["id"],
          "timestamp": row["timestamp"],
          "insulin": row["insulin"],
          "testosterone": row["testosterone"],
          "tsh": row["tsh"],
          "prolactin": row["prolactin"],
          "lh": row["lh"],
          "fsh": row["fsh"],
          "dheas": row["dheas"],
          "shbg": row["shbg"],
          "hba1c": row["hba1c"],
          "vitD": row["vitD"],
          "report": row["report"],
          "has_result": row["has_result"],
      })
    return {"status": "success", "history": history}
  except Exception as e:
    return {"status": "error", "message": str(e)}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
