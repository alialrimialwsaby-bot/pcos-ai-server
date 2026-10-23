
from datetime import datetime
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from medical_analyzer import router as medical_analyzer_router


# =========================================================
# Application configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "pcos_records.db"

app = FastAPI(
    title="PCOS & Universal Medical AI Clinical Intelligence Engine",
    version="1.0.0",
    description=(
        "Clinical support API for laboratory data and medical image reports."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the ultrasound analyzer routes from medical_analyzer.py.
# This router must define /analyze-ultrasound without another /api prefix
# if its APIRouter already uses prefix="/api".
app.include_router(medical_analyzer_router)


# =========================================================
# Database
# =========================================================

def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
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
            """
        )


init_db()


# =========================================================
# Laboratory input model
# =========================================================

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


# =========================================================
# Laboratory report engine
# Important: fixed thresholds are screening rules, not an
# independently validated AI model or a clinical diagnosis.
# =========================================================

class PCOSTheAIModel:

    def analyze_case(self, data: PatientInput) -> Dict[str, Any]:
        risk_scores = {
            "insulin_resistance": 0,
            "androgen_markers": 0,
            "dhea_s": 0,
            "thyroid_prolactin": 0,
            "vitamin_d": 0,
        }

        findings = []
        entered_any_value = False

        # Check that at least one non-zero value was entered.
        values = data.model_dump()
        entered_any_value = any(value != 0 for value in values.values())

        if not entered_any_value:
            raise HTTPException(
                status_code=422,
                detail=(
                    "أدخل قيمة فحص واحدة على الأقل. "
                    "الصفر يُعامل هنا كقيمة غير مدخلة."
                ),
            )

        # These rules are deliberately descriptive and provisional.
        # Lab interpretation depends on units, reference ranges,
        # age, clinical context, and laboratory methods.

        if data.hba1c > 0:
            if data.hba1c >= 6.5:
                findings.append(
                    "HbA1c: القيمة المدخلة تقع ضمن نطاق يُستخدم "
                    "للاشتباه بالسكري، ويجب تأكيدها وتقييمها طبيًا."
                )
                risk_scores["insulin_resistance"] += 2
            elif data.hba1c >= 5.7:
                findings.append(
                    "HbA1c: القيمة المدخلة تقع ضمن نطاق ما قبل السكري "
                    "وفق المعايير الشائعة، مع ضرورة التحقق من السياق الطبي."
                )
                risk_scores["insulin_resistance"] += 1

        if data.insulin > 0:
            findings.append(
                "Insulin: سُجّمت قيمة الإنسولين المدخلة. "
                "لا يمكن تحديد مقاومة الإنسولين من هذه القيمة وحدها "
                "دون الوحدات والسياق السريري."
            )

        if data.testosterone > 0:
            findings.append(
                "Testosterone: سُجّمت قيمة التستوستيرون المدخلة. "
                "يلزم التحقق من الوحدة والنطاق المرجعي والعمر "
                "قبل تفسيرها."
            )

        if data.lh > 0 or data.fsh > 0:
            findings.append(
                "LH/FSH: سُجّمت قيم الهرمونات المدخلة. "
                "يتطلب تفسيرها معرفة الوحدات ومرحلة الدورة الشهرية "
                "والسياق السريري؛ ولا تكفي وحدها لتشخيص PCOS."
            )

        if data.dheas > 0:
            findings.append(
                "DHEA-S: سُجّمت قيمة الفحص. "
                "يجب مقارنتها بالنطاق المرجعي المناسب للعمر والجنس "
                "والوحدة المستخدمة."
            )

        if data.tsh > 0:
            findings.append(
                "TSH: سُجّمت قيمة الفحص. "
                "يجب تفسيرها وفق النطاق المرجعي للمختبر والسياق الطبي."
            )

        if data.prolactin > 0:
            findings.append(
                "Prolactin: سُجّمت قيمة الفحص. "
                "يجب تفسيرها وفق الوحدة والنطاق المرجعي وظروف إجراء الفحص."
            )

        if data.shbg > 0:
            findings.append(
                "SHBG: سُجّمت قيمة الفحص. "
                "يلزم التحقق من الوحدة والنطاق المرجعي قبل تفسيرها."
            )

        if data.vitD > 0:
            findings.append(
                "Vitamin D: سُجّمت قيمة الفحص. "
                "يعتمد تفسيرها على نوع الفحص والوحدة والنطاق المرجعي."
            )

        report = (
            "تقرير مراجعة الفحوصات المخبرية\n"
            f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "القيم والملاحظات:\n"
            + "\n".join(f"- {item}" for item in findings)
            + "\n\n"
            "حدود التقرير:\n"
            "- هذا التقرير آلي ومبدئي وليس تشخيصًا طبيًا.\n"
            "- لم تُتحقق وحدات القياس أو النطاقات المرجعية تلقائيًا.\n"
            "- لا تُستنتج متلازمة تكيس المبايض من فحص منفرد.\n"
            "- ينبغي مراجعة النتائج مع مختص صحي."
        )

        # Save the laboratory report using the existing database schema.
        try:
            with get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO patient_records (
                        timestamp, insulin, testosterone, tsh, prolactin,
                        lh, fsh, dheas, shbg, hba1c, vitD, report, has_result
                    )
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
                        1 if findings else 0,
                    ),
                )
                record_id = cursor.lastrowid

        except sqlite3.Error:
            raise HTTPException(
                status_code=500,
                detail="تعذر حفظ التقرير في قاعدة البيانات.",
            )

        return {
            "status": "success",
            "record_id": record_id,
            "hasResult": bool(findings),
            "report": report,
            "ai_confidence_scores": risk_scores,
            "is_diagnostic": False,
        }


ai_model = PCOSTheAIModel()


# =========================================================
# API: laboratory analysis
# POST /api/analyze
# =========================================================

@app.post("/api/analyze")
def analyze_patient(data: PatientInput):
    return ai_model.analyze_case(data)


# =========================================================
# API: recent report history
# GET /api/history
# =========================================================

@app.get("/api/history")
def get_patient_history():
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM patient_records
                ORDER BY id DESC
                LIMIT 20
                """
            ).fetchall()

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

        return {
            "status": "success",
            "count": len(history),
            "history": history,
        }

    except sqlite3.Error:
        raise HTTPException(
            status_code=500,
            detail="تعذر جلب سجل التقارير.",
        )


# =========================================================
# Health check
# =========================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PCOS & Universal Medical AI Clinical Intelligence Engine",
        "docs": "/docs",
        "ultrasound_endpoint": "/api/analyze-ultrasound",
    }


# =========================================================
# Local startup
# Railway uses its configured start command / Procfile.
# =========================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
